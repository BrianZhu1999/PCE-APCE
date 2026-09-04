from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


METHOD_ORDER = [
    "DEnKF",
    "LETKF",
    "EnSF",
    "IEnSF",
    "EnSF-LR",
    "EnSF-LR-Ridge",
    "PCE",
    "APCE",
]
CASE_ORDER = ["Wave", "Spring", "Heat"]
METHOD_COLORS = {
    "DEnKF": "#7A7A7A",
    "LETKF": "#A0A0A0",
    "EnSF": "#9CBBD5",
    "IEnSF": "#4F83B4",
    "EnSF-LR": "#D8A37A",
    "EnSF-LR-Ridge": "#B8695B",
    "PCE": "#4E79A7",
    "APCE": "#E59F3A",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = dict(row)
            for key in (
                "nrmse",
                "crps",
                "coverage_90",
                "interval_width_90",
                "runtime_seconds",
                "max_abs_ratio",
            ):
                parsed[key] = float(str(row[key]))
            parsed["valid"] = str(row["valid"]).lower() == "true"
            parsed["case"] = str(row["case"]).capitalize()
            parsed["label"] = str(row["label"])
            rows.append(parsed)
    return rows


def append_wave_pce_rows(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["seed"] != "2026080600" or row["method"] not in {"pce", "apce"}:
                continue
            label = "PCE" if row["method"] == "pce" else "APCE"
            rows.append(
                {
                    "case": "Wave",
                    "method": row["method"],
                    "label": label,
                    "valid": True,
                    "validity_reason": "finite_and_bounded",
                    "nrmse": float(row["displacement_nrmse"]),
                    "crps": float(row["crps"]),
                    "coverage_90": float(row["coverage_90"]),
                    "interval_width_90": float(row["interval_width_90"]),
                    "runtime_seconds": float(row["runtime_seconds"]),
                    "max_abs_ratio": math.nan,
                }
            )


def append_spring_heat_pce_rows(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["seed"] != "2026080600" or row["method"] not in {"pce", "apce"}:
                continue
            label = "PCE" if row["method"] == "pce" else "APCE"
            rows.append(
                {
                    "case": row["case"].capitalize(),
                    "method": row["method"],
                    "label": label,
                    "valid": True,
                    "validity_reason": "finite_and_bounded",
                    "nrmse": float(row["nrmse"]),
                    "crps": float(row["crps"]),
                    "coverage_90": float(row["coverage_90"]),
                    "interval_width_90": float(row["interval_width_90"]),
                    "runtime_seconds": float(row["runtime_seconds"]),
                    "max_abs_ratio": math.nan,
                }
            )


def row_for(rows: list[dict[str, object]], case: str, method: str) -> dict[str, object]:
    for row in rows:
        if row["case"] == case and row["label"] == method:
            return row
    raise KeyError((case, method))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="normal",
    )


def status_code(row: dict[str, object]) -> int:
    if bool(row["valid"]) and float(row["nrmse"]) < 1.0:
        return 0
    if bool(row["valid"]):
        return 1
    return 2


def plot_status(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    matrix = np.zeros((len(CASE_ORDER), len(METHOD_ORDER)))
    for i, case in enumerate(CASE_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            matrix[i, j] = status_code(row_for(rows, case, method))
    cmap = ListedColormap(["#EAF2EA", "#F3E9C7", "#F1D5D0"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(CASE_ORDER)))
    ax.set_yticklabels(CASE_ORDER)
    ax.set_title("Admission status")
    ax.tick_params(length=0)
    for i, case in enumerate(CASE_ORDER):
        for j, method in enumerate(METHOD_ORDER):
            row = row_for(rows, case, method)
            text = "V" if status_code(row) == 0 else ("B" if status_code(row) == 1 else "F")
            ax.text(j, i, text, ha="center", va="center", fontsize=6.2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_grouped_bars(ax: plt.Axes, rows: list[dict[str, object]], metric: str, title: str, ylabel: str) -> None:
    x = np.arange(len(CASE_ORDER))
    width = 0.115
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2) * width
    for j, method in enumerate(METHOD_ORDER):
        values = [float(row_for(rows, case, method)[metric]) for case in CASE_ORDER]
        valid = [bool(row_for(rows, case, method)["valid"]) for case in CASE_ORDER]
        bars = ax.bar(
            x + offsets[j],
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.95,
            label=method,
        )
        for bar, ok in zip(bars, valid):
            if not ok:
                bar.set_hatch("//")
                bar.set_alpha(0.62)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(CASE_ORDER)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)


def plot_amplitude(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    x = np.arange(len(CASE_ORDER))
    width = 0.115
    amplitude_methods = ["DEnKF", "LETKF", "EnSF", "IEnSF", "EnSF-LR", "EnSF-LR-Ridge"]
    offsets = (np.arange(len(amplitude_methods)) - (len(amplitude_methods) - 1) / 2) * width
    for j, method in enumerate(amplitude_methods):
        values = [float(row_for(rows, case, method)["max_abs_ratio"]) for case in CASE_ORDER]
        valid = [bool(row_for(rows, case, method)["valid"]) for case in CASE_ORDER]
        bars = ax.bar(
            x + offsets[j],
            values,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.95,
        )
        for bar, ok in zip(bars, valid):
            if not ok:
                bar.set_hatch("//")
                bar.set_alpha(0.62)
    ax.axhline(100.0, color="#B33A3A", linewidth=0.9)
    ax.text(2.48, 100.0, "100x threshold", color="#B33A3A", ha="right", va="bottom", fontsize=6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(CASE_ORDER)
    ax.set_title("Baseline amplitude stability")
    ax.set_ylabel("max |state| / max |truth|")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)


def plot_calibration(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    markers = {"Wave": "o", "Spring": "s", "Heat": "^"}
    for case in CASE_ORDER:
        for method in METHOD_ORDER:
            row = row_for(rows, case, method)
            ok = bool(row["valid"]) and float(row["nrmse"]) < 1.0
            ax.scatter(
                float(row["coverage_90"]),
                float(row["interval_width_90"]),
                s=44 if ok else 36,
                marker=markers[case],
                facecolor=METHOD_COLORS[method],
                edgecolor="#333333",
                linewidth=0.45,
                alpha=0.95 if ok else 0.45,
            )
    ax.axvline(0.90, color="#555555", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlim(0.35, 1.03)
    ax.set_title("Calibration and spread")
    ax.set_xlabel("90% coverage")
    ax.set_ylabel("90% interval width")
    ax.grid(color="#D8D8D8", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    case_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[case],
            color="none",
            markerfacecolor="#F4F4F4",
            markeredgecolor="#333333",
            markersize=4.8,
            label=case,
        )
        for case in CASE_ORDER
    ]
    ax.legend(handles=case_handles, loc="upper left", bbox_to_anchor=(1.01, 1.02), title=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--wave-pce", type=Path, required=True)
    parser.add_argument("--spring-heat-pce", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    setup_style()
    rows = read_rows(args.input)
    append_wave_pce_rows(rows, args.wave_pce)
    append_spring_heat_pce_rows(rows, args.spring_heat_pce)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output_dir / "source_data_modern_baseline_admission.csv")
    shutil.copy2(args.wave_pce, args.output_dir / "source_data_wave_pce_apce_seed2026080600.csv")
    shutil.copy2(args.spring_heat_pce, args.output_dir / "source_data_spring_heat_pce_apce.csv")
    combined_fields = [
        "case",
        "method",
        "label",
        "valid",
        "validity_reason",
        "nrmse",
        "crps",
        "coverage_90",
        "interval_width_90",
        "runtime_seconds",
        "max_abs_ratio",
    ]
    with (args.output_dir / "source_data_combined_admission_with_pce_apce.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in combined_fields} for row in rows)

    fig = plt.figure(figsize=(7.2, 5.45), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.12, 1.08, 1.08],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1:])

    plot_status(ax_a, rows)
    plot_grouped_bars(ax_b, rows, "nrmse", "State error", "nRMSE")
    plot_grouped_bars(ax_c, rows, "crps", "Probabilistic error", "CRPS")
    plot_amplitude(ax_d, rows)
    plot_calibration(ax_e, rows)

    for label, ax in zip("abcde", [ax_a, ax_b, ax_c, ax_d, ax_e], strict=True):
        add_panel_label(ax, label)

    handles = [
        Patch(facecolor=METHOD_COLORS[method], edgecolor="#333333", linewidth=0.35, label=method)
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        ncol=8,
        handlelength=1.4,
        columnspacing=0.72,
    )
    base = args.output_dir / "figure_modern_baseline_admission"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
