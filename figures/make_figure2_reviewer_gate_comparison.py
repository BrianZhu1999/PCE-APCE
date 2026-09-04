from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

OUT_BASE = FIGURE_ROOT / "figure2_reviewer_gate_fullest_comparison_v1"


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

METHOD_ORDER = [
    "denkf",
    "letkf",
    "iensf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
    "pce_refined",
    "apce_refined",
    "oracle_alpha",
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
    "denkf": "#B4C0E4",
    "letkf": "#7884B4",
    "iensf": "#484878",
    "aug_enkf": "#9A4D8E",
    "bma_static": "#42949E",
    "pce": "#F0C0CC",
    "apce": "#E85C2A",
    "pce_refined": "#F6CFCB",
    "apce_refined": "#B64342",
    "oracle_alpha": "#272727",
}

METRICS = [
    ("nrmse", "nRMSE", "lower"),
    ("crps", "CRPS", "lower"),
    ("coverage_90", "90% coverage", "target"),
    ("interval_width_90", "Interval width", "lower"),
    ("alpha_absolute_error", r"$\alpha$ error", "lower"),
]


def read_rows(path: Path, source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("case") not in CASES:
                continue
            method = str(row.get("method", "")).strip()
            if method not in METHOD_ORDER:
                continue
            parsed: dict[str, object] = {
                "case": row["case"],
                "method": method,
                "source": source,
            }
            seed_text = row.get("seed", "")
            try:
                seed_int = int(float(seed_text))
            except ValueError:
                seed_int = -1
            parsed["seed"] = seed_int
            parsed["paired_seed"] = seed_int % 100 if seed_int >= 100 else seed_int
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


def rows_for(rows: list[dict[str, object]], case: str, method: str) -> list[dict[str, object]]:
    return [r for r in rows if r["case"] == case and r["method"] == method]


def clean_values(group: list[dict[str, object]], metric: str) -> np.ndarray:
    values = np.asarray([float(r[metric]) for r in group], dtype=float)
    return values[np.isfinite(values)]


def source_kind(group: list[dict[str, object]]) -> str:
    sources = {str(r["source"]) for r in group}
    if sources == {"formal50"}:
        return "formal50"
    if sources == {"smoke5"}:
        return "smoke5"
    return "+".join(sorted(sources))


def sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def improve_score(method_mean: float, reference_mean: float, metric_kind: str) -> float:
    if not np.isfinite(method_mean) or not np.isfinite(reference_mean):
        return math.nan
    if metric_kind == "target":
        # Score is closeness to 90% coverage relative to APCE.
        ref_gap = abs(reference_mean - 0.90)
        method_gap = abs(method_mean - 0.90)
        return ref_gap - method_gap
    return method_mean - reference_mean


def fmt_tick(x: float) -> str:
    if abs(x) >= 1:
        return f"{x:g}"
    if abs(x) >= 0.01:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.1e}"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def draw_box_panel(
    ax,
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    metric_kind: str,
) -> None:
    x_positions: list[float] = []
    values_by_pos: list[np.ndarray] = []
    colors_by_pos: list[str] = []
    hatch_by_pos: list[str] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    method_positions: dict[tuple[str, str], float] = {}
    centers: list[float] = []
    width = 0.42
    gap_case = 0.95
    gap_method = 0.10
    xpos = 0.0
    for case in CASES:
        case_positions = []
        for method in METHOD_ORDER:
            group = rows_for(rows, case, method)
            values = clean_values(group, metric)
            if values.size == 0:
                continue
            x_positions.append(xpos)
            values_by_pos.append(values)
            colors_by_pos.append(METHOD_COLORS[method])
            hatch_by_pos.append("//" if source_kind(group) == "smoke5" else "")
            method_positions[(case, method)] = xpos
            case_positions.append(xpos)
            tick_positions.append(xpos)
            tick_labels.append(METHOD_LABELS[method])
            xpos += width + gap_method
        if case_positions:
            center = 0.5 * (case_positions[0] + case_positions[-1])
            centers.append(center)
            ax.text(
                center,
                1.03,
                CASE_LABELS[case],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
            )
            xpos += gap_case

    bplot = ax.boxplot(
        values_by_pos,
        positions=x_positions,
        widths=width * 0.80,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#272727", "linewidth": 0.8},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.7},
        capprops={"color": "#4D4D4D", "linewidth": 0.7},
    )
    for patch, color, hatch in zip(bplot["boxes"], colors_by_pos, hatch_by_pos):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#272727")
        patch.set_linewidth(0.7)
        patch.set_hatch(hatch)

    rng = np.random.default_rng(20260810)
    for xp, vals, color in zip(x_positions, values_by_pos, colors_by_pos):
        jitter = rng.normal(0, width * 0.045, size=vals.size)
        ax.scatter(
            xp + jitter,
            vals,
            s=8 if vals.size <= 5 else 5,
            color=color,
            edgecolor="#272727",
            linewidth=0.25,
            alpha=0.85 if vals.size <= 5 else 0.35,
            zorder=4,
        )

    if metric == "coverage_90":
        ax.axhline(0.90, color="#767676", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_ylabel(title, fontsize=9)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=55, ha="right", fontsize=6.8)
    ax.tick_params(axis="y", labelsize=8, length=2.5, width=0.7)
    ax.tick_params(axis="x", length=0)
    ax.grid(False)
    ymin, ymax = ax.get_ylim()
    pad = 0.08 * (ymax - ymin)
    if metric == "coverage_90":
        ax.set_ylim(max(0.0, ymin - pad * 0.2), min(1.05, ymax + pad))
    else:
        ax.set_ylim(max(0.0, ymin - pad * 0.2), ymax + pad)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_tick(x)))


def draw_gain_panel(
    ax,
    rows: list[dict[str, object]],
    metric: str,
    title: str,
    metric_kind: str,
    reference: str,
) -> None:
    competitors = [
        "denkf",
        "letkf",
        "iensf",
        "aug_enkf",
        "bma_static",
        "pce",
        "pce_refined",
    ]
    if reference == "apce_refined":
        competitors = [
            "denkf",
            "letkf",
            "iensf",
            "aug_enkf",
            "bma_static",
            "pce",
            "apce",
            "pce_refined",
        ]
    y_labels: list[str] = []
    estimates: list[float] = []
    errors: list[float] = []
    colors: list[str] = []
    markers: list[str] = []
    for case in CASES:
        ref_group = rows_for(rows, case, reference)
        ref_values = clean_values(ref_group, metric)
        if ref_values.size == 0:
            continue
        ref_mean = float(np.mean(ref_values))
        for method in competitors:
            group = rows_for(rows, case, method)
            vals = clean_values(group, metric)
            if vals.size == 0:
                continue
            if method == reference:
                continue
            gain_values = []
            # paired when the same seed coordinates exist; otherwise group-level smoke/full mean.
            for r in group:
                seed = int(r["paired_seed"])
                matched_ref = [
                    q
                    for q in ref_group
                    if int(q["paired_seed"]) == seed and np.isfinite(float(q[metric]))
                ]
                if matched_ref and np.isfinite(float(r[metric])):
                    gain_values.append(float(r[metric]) - float(matched_ref[0][metric]))
            if not gain_values:
                gain_values = [improve_score(float(v), ref_mean, metric_kind) for v in vals]
            gain = float(np.mean(gain_values))
            estimates.append(gain)
            errors.append(sem(np.asarray(gain_values, dtype=float)))
            y_labels.append(f"{CASE_LABELS[case]} · {METHOD_LABELS[method]}")
            colors.append(METHOD_COLORS[method])
            markers.append("s" if source_kind(group) == "smoke5" else "o")

    y = np.arange(len(y_labels))[::-1]
    for yi, est, err, color, marker in zip(y, estimates, errors, colors, markers):
        ax.errorbar(
            est,
            yi,
            xerr=err,
            fmt=marker,
            ms=4.2,
            color=color,
            markeredgecolor="#272727",
            markeredgewidth=0.4,
            elinewidth=0.8,
            capsize=2.0,
            zorder=3,
        )
    ax.axvline(0.0, color="#767676", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=6.8)
    xlabel = f"{title}: comparator − {METHOD_LABELS[reference]}"
    if metric_kind == "target":
        xlabel = f"{title}: coverage-closeness gain vs {METHOD_LABELS[reference]}"
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(f"{METHOD_LABELS[reference]} advantage map", fontsize=9, pad=8)
    ax.tick_params(axis="x", labelsize=8, length=2.5, width=0.7)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    values = np.asarray(estimates, dtype=float)
    if values.size:
        lo, hi = np.nanmin(values), np.nanmax(values)
        span = max(hi - lo, 1e-6)
        ax.set_xlim(lo - 0.16 * span, hi + 0.16 * span)


def draw_n_matrix(ax, rows: list[dict[str, object]]) -> None:
    matrix = np.full((len(CASES), len(METHOD_ORDER)), np.nan)
    smoke_mask = np.zeros_like(matrix, dtype=bool)
    for i, case in enumerate(CASES):
        for j, method in enumerate(METHOD_ORDER):
            group = rows_for(rows, case, method)
            if group:
                matrix[i, j] = len(group)
                smoke_mask[i, j] = source_kind(group) == "smoke5"
    cmap = matplotlib.colors.ListedColormap(["#F0E0D0", "#E0E0F0", "#E0F0F0"])
    display = np.where(np.isnan(matrix), np.nan, np.where(smoke_mask, 1, 2))
    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(len(CASES)):
        for j in range(len(METHOD_ORDER)):
            if np.isfinite(matrix[i, j]):
                label = f"n={int(matrix[i, j])}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#272727",
                )
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(CASES)))
    ax.set_yticklabels([CASE_LABELS[c] for c in CASES], fontsize=8)
    ax.set_title("Available paired seeds", fontsize=9, pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_summary(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "method",
        "label",
        "source",
        "n",
        "nrmse_mean",
        "nrmse_sem",
        "crps_mean",
        "crps_sem",
        "coverage_90_mean",
        "coverage_90_sem",
        "interval_width_90_mean",
        "interval_width_90_sem",
        "alpha_absolute_error_mean",
        "alpha_absolute_error_sem",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            for method in METHOD_ORDER:
                group = rows_for(rows, case, method)
                if not group:
                    continue
                row = {
                    "case": case,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "source": source_kind(group),
                    "n": len(group),
                }
                for metric, _, _ in METRICS:
                    vals = clean_values(group, metric)
                    if vals.size:
                        row[f"{metric}_mean"] = float(np.mean(vals))
                        row[f"{metric}_sem"] = sem(vals)
                    else:
                        row[f"{metric}_mean"] = ""
                        row[f"{metric}_sem"] = ""
                writer.writerow(row)


def main() -> None:
    rows = load_rows()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    save_summary(rows, SOURCE_ROOT / "figure2_reviewer_gate_fullest_comparison_summary_v1.csv")

    fig = plt.figure(figsize=(13.8, 9.1))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.15, 1.15, 1.05],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.055,
        right=0.985,
        top=0.925,
        bottom=0.105,
        hspace=0.72,
        wspace=0.34,
    )
    axes = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[0, 2]),
        "d": fig.add_subplot(gs[1, 0]),
        "e": fig.add_subplot(gs[1, 1]),
        "f": fig.add_subplot(gs[1, 2]),
        "g": fig.add_subplot(gs[2, 0]),
        "h": fig.add_subplot(gs[2, 1]),
        "i": fig.add_subplot(gs[2, 2]),
    }

    draw_box_panel(axes["a"], rows, "nrmse", "nRMSE", "lower")
    draw_box_panel(axes["b"], rows, "crps", "CRPS", "lower")
    draw_box_panel(axes["c"], rows, "coverage_90", "90% coverage", "target")
    draw_box_panel(axes["d"], rows, "interval_width_90", "Interval width", "lower")
    draw_box_panel(axes["e"], rows, "alpha_absolute_error", r"$\alpha$ error", "lower")
    draw_n_matrix(axes["f"], rows)
    draw_gain_panel(axes["g"], rows, "nrmse", "nRMSE", "lower", "apce")
    draw_gain_panel(axes["h"], rows, "crps", "CRPS", "lower", "apce")
    draw_gain_panel(axes["i"], rows, "nrmse", "nRMSE", "lower", "apce_refined")

    for label, ax in axes.items():
        add_panel_label(ax, label)

    legend_handles = [
        Patch(facecolor=METHOD_COLORS[m], edgecolor="#272727", label=METHOD_LABELS[m], alpha=0.72)
        for m in METHOD_ORDER
    ]
    source_handles = [
        Patch(facecolor="#D8D8D8", edgecolor="#272727", label="formal n=50"),
        Patch(facecolor="#D8D8D8", edgecolor="#272727", hatch="//", label="smoke n=5"),
        Line2D([0], [0], color="#767676", linestyle="--", linewidth=0.9, label="reference/target"),
    ]
    fig.legend(
        handles=legend_handles + source_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=13,
        fontsize=8.4,
        handlelength=1.1,
        columnspacing=0.9,
        handletextpad=0.35,
        frameon=False,
    )

    fig.text(
        0.055,
        0.035,
        "Full formal seeds are used for DEnKF, LETKF, IEnSF, PCE, APCE and Oracle-alpha (n=50 per case); reviewer-risk baselines and continuous-refinement variants currently use smoke seeds (n=5 per case). Boxes show seed distributions with overlaid paired-seed points.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
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
