from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "ncs_chinese_submission" / "source_data"
FIGURE_ROOT = PROJECT_ROOT / "ncs_chinese_submission" / "figures"

FORMAL = SOURCE_ROOT / "figure2_run_source_data_20260807.csv"
REVIEWER = (
    SOURCE_ROOT
    / "figure2_reviewer_gate_20260810_4gpu"
    / "figure2_reviewer_gate_new_method_runs_20260810.csv"
)
OUT_BASE = FIGURE_ROOT / "figure2_reviewer_gate_fullest_comparison_v2_matrix"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 9
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

CASES = ["wave", "spring", "heat"]
CASE_LABELS = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}

METHODS = [
    "oracle_alpha",
    "apce_refined",
    "pce_refined",
    "apce",
    "pce",
    "bma_static",
    "aug_enkf",
    "iensf",
    "letkf",
    "denkf",
]
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
    "pce_refined": "PCE-refined",
    "apce_refined": "APCE-refined",
    "oracle_alpha": "Oracle",
}
METHOD_COLORS = {
    "oracle_alpha": "#272727",
    "apce_refined": "#B64342",
    "pce_refined": "#F6CFCB",
    "apce": "#E85C2A",
    "pce": "#F0C0CC",
    "bma_static": "#42949E",
    "aug_enkf": "#9A4D8E",
    "iensf": "#484878",
    "letkf": "#7884B4",
    "denkf": "#B4C0E4",
}

METRICS = [
    ("nrmse", "nRMSE", "low"),
    ("crps", "CRPS", "low"),
    ("coverage_90", "90% coverage", "target"),
    ("interval_width_90", "Interval width", "low"),
    ("alpha_absolute_error", r"$\alpha$ error", "low"),
]

LOW_GOOD_CMAP = LinearSegmentedColormap.from_list(
    "low_good", ["#E0F0F0", "#F0E0D0", "#E9A6A1", "#B64342"]
)
TARGET_CMAP = LinearSegmentedColormap.from_list(
    "target_good", ["#AADCA9", "#DDF3DE", "#F0E0D0", "#E9A6A1", "#B64342"]
)


def read_rows(path: Path, source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case = row.get("case", "")
            method = row.get("method", "")
            if case not in CASES or method not in METHODS:
                continue
            parsed: dict[str, object] = {"case": case, "method": method, "source": source}
            try:
                seed = int(float(row.get("seed", "-1")))
            except ValueError:
                seed = -1
            parsed["seed"] = seed
            parsed["paired_seed"] = seed % 100 if seed >= 100 else seed
            for metric, _, _ in METRICS:
                try:
                    parsed[metric] = float(row.get(metric, "nan"))
                except ValueError:
                    parsed[metric] = math.nan
            rows.append(parsed)
    return rows


def load_rows() -> list[dict[str, object]]:
    rows = read_rows(FORMAL, "formal50")
    rows.extend(read_rows(REVIEWER, "smoke5"))
    return rows


def get_group(rows: list[dict[str, object]], method: str, case: str) -> list[dict[str, object]]:
    return [r for r in rows if r["method"] == method and r["case"] == case]


def values_for(rows: list[dict[str, object]], method: str, case: str, metric: str) -> np.ndarray:
    vals = np.asarray([float(r[metric]) for r in get_group(rows, method, case)], dtype=float)
    return vals[np.isfinite(vals)]


def source_for(rows: list[dict[str, object]], method: str, case: str) -> str:
    group = get_group(rows, method, case)
    sources = {str(r["source"]) for r in group}
    if sources == {"formal50"}:
        return "formal50"
    if sources == {"smoke5"}:
        return "smoke5"
    if not sources:
        return "none"
    return "+".join(sorted(sources))


def mean_matrix(rows: list[dict[str, object]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(METHODS), len(CASES)), np.nan)
    n_matrix = np.zeros((len(METHODS), len(CASES)), dtype=int)
    for i, method in enumerate(METHODS):
        for j, case in enumerate(CASES):
            vals = values_for(rows, method, case, metric)
            if vals.size:
                matrix[i, j] = float(np.mean(vals))
                n_matrix[i, j] = int(vals.size)
    return matrix, n_matrix


def score_matrix(raw: np.ndarray, metric_kind: str) -> np.ndarray:
    if metric_kind == "target":
        score = np.abs(raw - 0.90)
    else:
        score = raw.copy()
    return score


def fmt_value(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return ""
    if metric == "coverage_90":
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.1e}"
    if abs(value) < 0.01:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if abs(value) < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def luminance(rgb) -> float:
    r, g, b = rgb[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def draw_heatmap_panel(
    ax,
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    metric_kind: str,
    show_ylabels: bool,
) -> None:
    raw, n_matrix = mean_matrix(rows, metric)
    score = score_matrix(raw, metric_kind)
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanpercentile(finite, 97))
        if vmax <= vmin:
            vmax = vmin + 1.0
    cmap = TARGET_CMAP if metric_kind == "target" else LOW_GOOD_CMAP
    im = ax.imshow(score, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i, method in enumerate(METHODS):
        for j, case in enumerate(CASES):
            value = raw[i, j]
            if not np.isfinite(value):
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F7F7F7", edgecolor="#D0D0D0", linewidth=0.4)
                ax.add_patch(rect)
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#767676")
                continue
            rgba = cmap((score[i, j] - vmin) / max(vmax - vmin, 1.0e-12))
            text_color = "white" if luminance(rgba) < 0.50 else "#272727"
            ax.text(j, i, fmt_value(value, metric), ha="center", va="center", fontsize=7.4, color=text_color)
            if source_for(rows, method, case) == "smoke5":
                hatch_rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor="#272727",
                    linewidth=0.45,
                    hatch="///",
                )
                ax.add_patch(hatch_rect)
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="white", linewidth=0.65))
            if method in {"apce", "apce_refined"}:
                ax.add_patch(
                    Rectangle(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        facecolor="none",
                        edgecolor="#B64342",
                        linewidth=0.9,
                    )
                )

    ax.set_xticks(np.arange(len(CASES)))
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES], fontsize=8)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS] if show_ylabels else [], fontsize=8)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=6.8, length=2.2, width=0.6)
    if metric_kind == "target":
        cbar.set_label("gap to 0.90", fontsize=6.8, labelpad=2)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_value(x, metric)))
    else:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_value(x, metric)))
    cbar.outline.set_linewidth(0.5)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, reps: int = 2000) -> tuple[float, float, float]:
    if a.size == 0 or b.size == 0:
        return math.nan, math.nan, math.nan
    n = min(a.size, b.size)
    diff = a[:n] - b[:n]
    mean = float(np.mean(diff))
    if n <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(20260810 + n)
    idx = rng.integers(0, n, size=(reps, n))
    boot = diff[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def draw_gain_table(ax, rows: list[dict[str, object]]) -> None:
    ax.axis("off")
    comparisons = [
        ("wave", "apce", "bma_static"),
        ("wave", "apce", "aug_enkf"),
        ("spring", "apce", "bma_static"),
        ("spring", "apce_refined", "bma_static"),
        ("heat", "apce", "bma_static"),
        ("heat", "apce_refined", "bma_static"),
    ]
    y0 = 0.92
    ax.text(0.00, y0, "Key paired gaps", fontsize=9, ha="left", va="center")
    ax.text(0.48, y0, "nRMSE", fontsize=8, ha="center", va="center", color="#4D4D4D")
    ax.text(0.72, y0, "CRPS", fontsize=8, ha="center", va="center", color="#4D4D4D")
    ax.text(0.94, y0, "n", fontsize=8, ha="center", va="center", color="#4D4D4D")
    for k, (case, ref, comp) in enumerate(comparisons):
        y = y0 - 0.12 * (k + 1)
        ax.text(
            0.00,
            y,
            f"{CASE_LABELS[case]}: {METHOD_LABELS[ref]} vs {METHOD_LABELS[comp]}",
            fontsize=7.6,
            ha="left",
            va="center",
            color="#272727",
        )
        ref_group = get_group(rows, ref, case)
        comp_group = get_group(rows, comp, case)
        n = min(len(ref_group), len(comp_group))
        for x, metric in [(0.48, "nrmse"), (0.72, "crps")]:
            ref_vals = np.asarray([float(r[metric]) for r in ref_group if np.isfinite(float(r[metric]))])
            comp_vals = np.asarray([float(r[metric]) for r in comp_group if np.isfinite(float(r[metric]))])
            mean, lo, hi = bootstrap_mean_diff(comp_vals, ref_vals)
            # positive comparator-reference means reference is better for lower-is-better metrics.
            color = "#2E9E44" if mean > 0 else "#B64342"
            ax.text(x, y, f"{mean:+.3g}", fontsize=7.6, ha="center", va="center", color=color)
        ax.text(0.94, y, str(n), fontsize=7.6, ha="center", va="center", color="#4D4D4D")
    ax.text(
        0.00,
        0.05,
        "Positive gap means the named APCE/APCE-refined variant has lower error than the comparator.",
        fontsize=7.2,
        ha="left",
        va="bottom",
        color="#606060",
    )


def draw_rank_panel(ax, rows: list[dict[str, object]]) -> None:
    rank_mat = np.full((len(METHODS), len(CASES)), np.nan)
    for j, case in enumerate(CASES):
        means = []
        for method in METHODS:
            vals = values_for(rows, method, case, "nrmse")
            means.append(float(np.mean(vals)) if vals.size else math.nan)
        order = np.argsort(np.asarray(means))
        ranks = np.empty(len(METHODS))
        ranks[:] = np.nan
        valid_rank = 1
        for idx in order:
            if np.isfinite(means[idx]):
                ranks[idx] = valid_rank
                valid_rank += 1
        rank_mat[:, j] = ranks
    im = ax.imshow(rank_mat, cmap=LinearSegmentedColormap.from_list("rank", ["#AADCA9", "#F0E0D0", "#E9A6A1"]), vmin=1, vmax=10, aspect="auto")
    for i, method in enumerate(METHODS):
        for j, case in enumerate(CASES):
            if np.isfinite(rank_mat[i, j]):
                ax.text(j, i, f"{int(rank_mat[i,j])}", ha="center", va="center", fontsize=7.4, color="#272727")
    ax.set_title("nRMSE rank", fontsize=9, pad=6)
    ax.set_xticks(np.arange(len(CASES)))
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES], fontsize=8)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("rank", fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6.8, length=2.2, width=0.6)
    cbar.outline.set_linewidth(0.5)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def save_summary(rows: list[dict[str, object]]) -> None:
    out = SOURCE_ROOT / "figure2_reviewer_gate_fullest_comparison_matrix_summary_v2.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["method", "case", "source", "n"] + [f"{metric}_mean" for metric, _, _ in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            for case in CASES:
                group = get_group(rows, method, case)
                if not group:
                    continue
                row = {"method": method, "case": case, "source": source_for(rows, method, case), "n": len(group)}
                for metric, _, _ in METRICS:
                    vals = values_for(rows, method, case, metric)
                    row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else ""
                writer.writerow(row)


def main() -> None:
    rows = load_rows()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    save_summary(rows)

    fig = plt.figure(figsize=(11.2, 7.7))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.72],
        width_ratios=[1.06, 1.06, 1.06, 1.06],
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.09,
        hspace=0.44,
        wspace=0.46,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2:]),
    ]

    draw_heatmap_panel(axes[0], rows, "nrmse", "nRMSE", "low", True)
    draw_heatmap_panel(axes[1], rows, "crps", "CRPS", "low", False)
    draw_heatmap_panel(axes[2], rows, "coverage_90", "90% coverage", "target", False)
    draw_heatmap_panel(axes[3], rows, "interval_width_90", "Interval width", "low", False)
    draw_heatmap_panel(axes[4], rows, "alpha_absolute_error", r"$\alpha$ error", "low", True)
    draw_rank_panel(axes[5], rows)
    draw_gain_table(axes[6], rows)

    for label, ax in zip("abcdefg", axes):
        add_panel_label(ax, label)

    handles = [
        Patch(facecolor="#E0F0F0", edgecolor="#272727", label="formal n=50"),
        Patch(facecolor="#E0F0F0", edgecolor="#272727", hatch="///", label="smoke n=5"),
        Patch(facecolor="none", edgecolor="#B64342", linewidth=1.0, label="APCE/APCE-refined"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.972),
        ncol=3,
        fontsize=8.5,
        handlelength=1.5,
        columnspacing=1.4,
        frameon=False,
    )
    fig.text(
        0.075,
        0.035,
        "Cells report seed means. Formal methods use the full 50 paired seeds; reviewer-risk baselines and continuous-refinement variants currently use 5 paired smoke seeds. Lower is better except coverage, where closeness to 0.90 is better.",
        fontsize=7.6,
        color="#4D4D4D",
        ha="left",
        va="bottom",
    )

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(f"{OUT_BASE}.{ext}", **kwargs)
    plt.close(fig)
    print(OUT_BASE)


if __name__ == "__main__":
    main()
