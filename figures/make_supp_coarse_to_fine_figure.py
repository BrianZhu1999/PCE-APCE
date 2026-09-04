from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CASES = ("Wave", "Spring", "Heat")
METHODS = ("PCE", "APCE")
COLORS = {
    "ink": "#272727",
    "neutral": "#767676",
    "neutral_light": "#C9CED3",
    "neutral_pale": "#EEF1F3",
    "blue": "#2E6FA3",
    "orange": "#E05A32",
    "teal": "#23857F",
}
METHOD_COLORS = {"PCE": COLORS["blue"], "APCE": COLORS["orange"]}
CASE_MARKERS = {"Wave": "o", "Spring": "s", "Heat": "^"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def organize(rows: list[dict[str, str]]) -> dict[str, Any]:
    seed_rows = [row for row in rows if row["record_type"] == "seed"]
    summary_rows = [row for row in rows if row["record_type"] == "summary"]
    data: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for case in CASES:
        for method in METHODS:
            selected = [row for row in seed_rows if row["case_label"] == case and row["method"] == method]
            if len(selected) != 50:
                raise ValueError(f"Expected 50 seed records for {case}/{method}")
            selected.sort(key=lambda row: int(row["seed"]))
            data[case][method]["seed"] = selected
            for metric in ("nrmse_log2_ratio", "alpha_error_change"):
                matches = [
                    row
                    for row in summary_rows
                    if row["case_label"] == case and row["method"] == method and row[metric]
                ]
                if len(matches) != 1:
                    raise ValueError(f"Missing summary for {case}/{method}/{metric}")
                data[case][method][metric] = matches[0]
    return data


def paired_stage_panel(ax: plt.Axes, data: dict[str, Any], metric: str, title: str, ylabel: str) -> None:
    positions = np.arange(len(CASES), dtype=float)
    offsets = {"PCE": -0.11, "APCE": 0.11}
    jitter_pattern = np.linspace(-0.032, 0.032, 50)
    for case_index, case in enumerate(CASES):
        for method in METHODS:
            rows = data[case][method]["seed"]
            if metric == "nrmse":
                coarse = np.asarray([float(row["coarse_nrmse"]) for row in rows])
                refined = np.asarray([float(row["refined_nrmse"]) for row in rows])
            else:
                coarse = np.asarray([float(row["coarse_alpha_error"]) for row in rows])
                refined = np.asarray([float(row["refined_alpha_error"]) for row in rows])
            x0 = positions[case_index] + offsets[method] - 0.035
            x1 = positions[case_index] + offsets[method] + 0.035
            for index, (before, after) in enumerate(zip(coarse, refined)):
                jitter = jitter_pattern[index]
                ax.plot(
                    [x0 + jitter, x1 + jitter],
                    [before, after],
                    color=METHOD_COLORS[method],
                    lw=0.34,
                    alpha=0.17,
                    zorder=1,
                )
            ax.scatter(
                np.full(50, x0) + jitter_pattern,
                coarse,
                s=4,
                facecolor="white",
                edgecolor=METHOD_COLORS[method],
                linewidth=0.35,
                alpha=0.55,
                zorder=2,
            )
            ax.scatter(
                np.full(50, x1) + jitter_pattern,
                refined,
                s=4,
                color=METHOD_COLORS[method],
                linewidth=0,
                alpha=0.55,
                zorder=2,
            )
            ax.plot(
                [x0, x1],
                [np.mean(coarse), np.mean(refined)],
                color=METHOD_COLORS[method],
                lw=1.6,
                marker="o",
                markersize=3.2,
                markerfacecolor=["white", METHOD_COLORS[method]][0],
                zorder=4,
            )
            ax.scatter([x1], [np.mean(refined)], s=16, color=METHOD_COLORS[method], zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(CASES)
    ax.set_title(title, loc="left", pad=4)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.45, len(CASES) - 0.55)
    ax.grid(axis="y", color=COLORS["neutral_pale"], lw=0.5, zorder=0)


def joint_change_panel(ax: plt.Axes, data: dict[str, Any]) -> None:
    for case in CASES:
        for method in METHODS:
            rows = data[case][method]["seed"]
            x = np.asarray([float(row["nrmse_log2_ratio"]) for row in rows])
            y = np.asarray([float(row["alpha_error_change"]) for row in rows])
            ax.scatter(
                x,
                y,
                s=9,
                marker=CASE_MARKERS[case],
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.34,
                zorder=2,
            )
            ax.scatter(
                [x.mean()],
                [y.mean()],
                s=38,
                marker=CASE_MARKERS[case],
                facecolor=METHOD_COLORS[method],
                edgecolor=COLORS["ink"],
                linewidth=0.45,
                zorder=4,
            )
            x_summary = data[case][method]["nrmse_log2_ratio"]
            y_summary = data[case][method]["alpha_error_change"]
            x_mean = float(x_summary["mean"])
            y_mean = float(y_summary["mean"])
            ax.errorbar(
                x_mean,
                y_mean,
                xerr=np.asarray(
                    [[x_mean - float(x_summary["ci_low"])], [float(x_summary["ci_high"]) - x_mean]]
                ),
                yerr=np.asarray(
                    [[y_mean - float(y_summary["ci_low"])], [float(y_summary["ci_high"]) - y_mean]]
                ),
                fmt="none",
                ecolor=METHOD_COLORS[method],
                elinewidth=0.9,
                capsize=1.8,
                capthick=0.7,
                zorder=3,
            )
    ax.axvline(0, color=COLORS["neutral"], linestyle="--", lw=0.75)
    ax.axhline(0, color=COLORS["neutral"], linestyle="--", lw=0.75)
    ax.set_xlabel("state change\n" + r"$\log_2(\mathrm{nRMSE}_{fine}/\mathrm{nRMSE}_{coarse})$")
    ax.set_ylabel("cognitive-error change\n" + r"$|\hat\alpha_f-\alpha^\star|-|\hat\alpha_c-\alpha^\star|$")
    ax.set_title("State and cognitive effects", loc="left", pad=4)


def make_figure(rows: list[dict[str, str]], output_dir: Path) -> Path:
    data = organize(rows)
    fig = plt.figure(figsize=(7.2, 2.75))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.075,
        right=0.985,
        bottom=0.22,
        top=0.82,
        wspace=0.48,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    paired_stage_panel(axes[0], data, "nrmse", "State reconstruction", "nRMSE")
    paired_stage_panel(axes[1], data, "alpha", "Cognitive-coordinate error", r"absolute $\alpha$ error")
    joint_change_panel(axes[2], data)
    for label, ax in zip("abc", axes):
        add_panel_label(ax, label)

    method_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[method], lw=1.7, marker="o", markersize=3.5, label=method)
        for method in METHODS
    ]
    stage_handles = [
        plt.Line2D([0], [0], color=COLORS["ink"], marker="o", markerfacecolor="white", lw=0, markersize=4, label="coarse"),
        plt.Line2D([0], [0], color=COLORS["ink"], marker="o", lw=0, markersize=4, label="refined"),
    ]
    case_handles = [
        plt.Line2D([0], [0], color=COLORS["neutral"], marker=CASE_MARKERS[case], lw=0, markersize=4, label=case)
        for case in CASES
    ]
    fig.legend(
        handles=method_handles + stage_handles + case_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=7,
        columnspacing=0.8,
        handlelength=1.4,
    )
    base = output_dir / "supp_coarse_to_fine_validation"
    save_figure(fig, base)
    return base


def inspect_outputs(base: Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = base.with_suffix(suffix)
        details: dict[str, Any] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        if suffix in {".png", ".tiff"}:
            with Image.open(path) as image:
                details["pixels"] = list(image.size)
                details["dpi"] = [float(value) for value in image.info.get("dpi", (0, 0))]
        if suffix == ".svg":
            text = path.read_text(encoding="utf-8")
            details["editable_text_nodes"] = text.count("<text")
            if details["editable_text_nodes"] == 0:
                raise ValueError("SVG text was converted to paths")
        outputs[suffix.lstrip(".")] = details
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the formal coarse-to-fine validation figure.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "ncs_english_latex" / "source_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ncs_english_latex" / "figures-supplemental")
    args = parser.parse_args()

    set_style()
    source_path = args.source_dir / "supp_coarse_to_fine_validation_source_data.csv"
    manifest_path = args.source_dir / "supp_coarse_to_fine_validation_manifest.json"
    base = make_figure(read_csv(source_path), args.output_dir)
    outputs = inspect_outputs(base)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = outputs
    manifest.setdefault("software", {}).update(
        {"matplotlib": matplotlib.__version__, "pillow": Image.__version__, "platform": platform.platform()}
    )
    manifest["qa"] = {
        "backend_exclusive": "Python/matplotlib",
        "svg_text_editable": True,
        "raster_target_dpi": 600,
        "formal_seed_count_per_case_method": 50,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
