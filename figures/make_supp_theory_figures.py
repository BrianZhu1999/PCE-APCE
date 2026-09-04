from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "ink": "#272727",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
    "neutral_pale": "#F2F2F2",
    "blue": "#2E6FA3",
    "blue_light": "#DCEAF4",
    "teal": "#23857F",
    "teal_light": "#DCEEEB",
    "orange": "#E05A32",
    "orange_light": "#F6DDD3",
    "red": "#B64342",
    "red_light": "#F3D7D4",
    "gold": "#C98A1D",
}
METHOD_COLORS = {"BMA": COLORS["teal"], "PCE": COLORS["blue"], "APCE": COLORS["orange"]}
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "weight", ["#F7F7F7", "#DCEAF4", "#78A8C7", "#185A86"], N=256
)


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


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 7.0,
    fontweight: str = "regular",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=fontweight,
        linespacing=1.15,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["neutral"],
    connectionstyle: str = "arc3",
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.85,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_shadow_workflow(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "a", x=-0.018, y=1.00)

    rounded_box(
        ax,
        (0.015, 0.34),
        0.125,
        0.30,
        "Candidate\ndynamics\n$\\alpha_1,\\ldots,\\alpha_K$",
        facecolor=COLORS["neutral_pale"],
        edgecolor=COLORS["neutral"],
        fontweight="bold",
    )
    rounded_box(
        ax,
        (0.19, 0.34),
        0.135,
        0.30,
        "Paired path-\nconditioned\nforecast",
        facecolor="#EEF3F6",
        edgecolor=COLORS["blue"],
    )
    arrow(ax, (0.14, 0.49), (0.19, 0.49), color=COLORS["blue"])

    rounded_box(
        ax,
        (0.38, 0.63),
        0.125,
        0.22,
        "Shadow bank\n$Z_t^{(k)}$",
        facecolor=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
        fontweight="bold",
    )
    rounded_box(
        ax,
        (0.38, 0.14),
        0.125,
        0.22,
        "Analysis bank\n$X_t^{(k)}$",
        facecolor=COLORS["teal_light"],
        edgecolor=COLORS["teal"],
        fontweight="bold",
    )
    arrow(ax, (0.325, 0.52), (0.38, 0.72), color=COLORS["blue"])
    arrow(ax, (0.325, 0.46), (0.38, 0.25), color=COLORS["teal"])

    rounded_box(
        ax,
        (0.56, 0.63),
        0.12,
        0.22,
        "Predictive\nevidence\n$\\widehat{\\ell}_{k,t}$",
        facecolor="#EEF3F6",
        edgecolor=COLORS["blue"],
    )
    rounded_box(
        ax,
        (0.56, 0.14),
        0.12,
        0.22,
        "State\nanalysis",
        facecolor="#EEF5F3",
        edgecolor=COLORS["teal"],
    )
    arrow(ax, (0.505, 0.74), (0.56, 0.74), color=COLORS["blue"])
    arrow(ax, (0.505, 0.25), (0.56, 0.25), color=COLORS["teal"])

    observation = Circle((0.62, 0.49), radius=0.043, facecolor="white", edgecolor=COLORS["ink"], lw=0.8)
    ax.add_patch(observation)
    ax.text(0.62, 0.49, "$y_t$", ha="center", va="center", fontsize=7.2)
    arrow(ax, (0.62, 0.535), (0.62, 0.63), color=COLORS["ink"])
    arrow(ax, (0.62, 0.445), (0.62, 0.36), color=COLORS["ink"])

    rounded_box(
        ax,
        (0.74, 0.63),
        0.11,
        0.22,
        "Cumulative\nlogits\n$L_{k,t}$",
        facecolor=COLORS["orange_light"],
        edgecolor=COLORS["orange"],
    )
    arrow(ax, (0.68, 0.74), (0.74, 0.74), color=COLORS["orange"])
    rounded_box(
        ax,
        (0.89, 0.63),
        0.095,
        0.22,
        "Path\nweights\n$w_{k,t}$",
        facecolor="#F8E8E1",
        edgecolor=COLORS["orange"],
        fontweight="bold",
    )
    arrow(ax, (0.85, 0.74), (0.89, 0.74), color=COLORS["orange"])

    rounded_box(
        ax,
        (0.76, 0.14),
        0.19,
        0.22,
        "Weighted state estimate\nand analysis interval",
        facecolor="#EEF5F3",
        edgecolor=COLORS["teal"],
        fontweight="bold",
    )
    arrow(ax, (0.68, 0.25), (0.76, 0.25), color=COLORS["teal"])
    arrow(
        ax,
        (0.94, 0.63),
        (0.90, 0.36),
        color=COLORS["orange"],
        connectionstyle="arc3,rad=0.05",
    )
    ax.text(
        0.44,
        0.91,
        "evidence remains outside the analysis feedback loop",
        color=COLORS["blue"],
        fontsize=6.6,
        fontweight="bold",
        ha="center",
    )


def draw_error_budget(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "b")
    ax.set_title("Evidence error budget", loc="left", pad=4)
    rounded_box(ax, (0.13, 0.72), 0.74, 0.15, "Ideal predictive score   $\\ell^\\star_{k,t}$", facecolor="white", edgecolor=COLORS["ink"])
    rounded_box(ax, (0.13, 0.43), 0.74, 0.15, "Gaussian / composite surrogate   $\\ell^G$ or $\\ell^D$", facecolor=COLORS["blue_light"], edgecolor=COLORS["blue"])
    rounded_box(ax, (0.13, 0.14), 0.74, 0.15, "Finite-ensemble implementation   $\\widehat{\\ell}_{k,t}$", facecolor=COLORS["orange_light"], edgecolor=COLORS["orange"])
    arrow(ax, (0.50, 0.72), (0.50, 0.58), color=COLORS["blue"])
    arrow(ax, (0.50, 0.43), (0.50, 0.29), color=COLORS["orange"])
    ax.text(0.54, 0.65, "surrogate bias", ha="left", va="center", fontsize=6.2, color=COLORS["blue"])
    ax.text(0.54, 0.36, "estimation error", ha="left", va="center", fontsize=6.2, color=COLORS["orange"])
    ax.text(
        0.50,
        0.01,
        "$\\widehat{\\ell}-\\ell^\\star=(\\ell^{G/D}-\\ell^\\star)+(\\widehat{\\ell}-\\ell^{G/D})$",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=COLORS["ink"],
    )
    ax.text(0.98, 0.91, "analytical decomposition", ha="right", fontsize=5.8, color=COLORS["neutral"])


def draw_error_decomposition(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_panel_label(ax, "e")
    ax.set_title("State--cognition error bound", loc="left", pad=4)
    rounded_box(ax, (0.09, 0.70), 0.82, 0.14, "$E_{\\mathrm{filtering}}$   finite-state analysis", facecolor=COLORS["teal_light"], edgecolor=COLORS["teal"])
    rounded_box(ax, (0.09, 0.46), 0.82, 0.14, "$E_{\\mathrm{cognition}}$   weighted path spread", facecolor=COLORS["orange_light"], edgecolor=COLORS["orange"])
    rounded_box(ax, (0.09, 0.22), 0.82, 0.14, "$E_{\\mathrm{grid}}$   candidate approximation", facecolor=COLORS["neutral_pale"], edgecolor=COLORS["neutral"])
    ax.text(0.50, 0.65, "+", ha="center", va="center", fontsize=9, color=COLORS["ink"])
    ax.text(0.50, 0.41, "+", ha="center", va="center", fontsize=9, color=COLORS["ink"])
    ax.text(
        0.50,
        0.08,
        "$\\|\\widehat{x}_t-x_t^\\star\\|\\;\\leq\\;E_{\\mathrm{filtering}}+E_{\\mathrm{cognition}}+E_{\\mathrm{grid}}$",
        ha="center",
        va="center",
        fontsize=6.4,
    )
    ax.text(0.50, -0.055, "concentration reduces cognition error; the grid term vanishes on-grid", ha="center", fontsize=5.7, color=COLORS["neutral"], clip_on=False)


def theory_series(rows: list[dict[str, str]], panel: str, series: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["panel"] == panel and row["series"] == series]
    selected.sort(key=lambda row: float(row["x_value"]))
    return (
        np.asarray([float(row["x_value"]) for row in selected], dtype=float),
        np.asarray([float(row["y_value"]) for row in selected], dtype=float),
    )


def make_theory_figure(rows: list[dict[str, str]], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(7.2, 5.25))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.96,
        bottom=0.11,
        top=0.95,
        hspace=0.45,
        wspace=0.38,
    )
    ax_c = fig.add_subplot(grid[0, 0])
    ax_d = fig.add_subplot(grid[0, 1])
    ax_f = fig.add_subplot(grid[1, 0])
    ax_g = fig.add_subplot(grid[1, 1])

    add_panel_label(ax_c, "a")
    ax_c.set_title("Finite-time path identification", loc="left", pad=4)
    for label, color in (("K=7", COLORS["blue"]), ("K=12", COLORS["orange"])):
        x, y = theory_series(rows, "c", label)
        ax_c.plot(x, y, lw=1.6, color=color, label=label)
    ax_c.set_yscale("log")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(1.0e-4, 1.08)
    ax_c.set_xlabel(r"standardized separation $\Delta^2/(2V)$")
    ax_c.set_ylabel("misidentification bound")
    ax_c.legend(loc="upper right")

    add_panel_label(ax_d, "b")
    ax_d.set_title("Analysis-feedback contraction", loc="left", pad=4)
    c_value, xi_value = theory_series(rows, "d", "sufficient_contraction_boundary")
    ax_d.fill_between(c_value, 0, xi_value, color=COLORS["blue_light"], alpha=0.9)
    ax_d.fill_between(c_value, xi_value, 1, color=COLORS["red_light"], alpha=0.65)
    ax_d.plot(c_value, xi_value, color=COLORS["ink"], linestyle="--", lw=1.0)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel(r"common-gain contraction $c$")
    ax_d.set_ylabel(r"gain-mismatch term $\xi$")

    add_panel_label(ax_f, "c")
    ax_f.set_title("Entropy-floor projection", loc="left", pad=4)
    candidate_colors = [COLORS["orange"]] + ["#9FA8B2"] * 6
    for index in range(1, 8):
        gamma, weights = theory_series(rows, "f", f"candidate_{index}")
        ax_f.plot(gamma, weights, color=candidate_colors[index - 1], lw=1.5 if index == 1 else 0.75, alpha=1.0 if index == 1 else 0.7)
    ax_f.set_xlim(0, 1)
    ax_f.set_ylim(0, 0.56)
    ax_f.set_xlabel(r"uniform mixing $\gamma$")
    ax_f.set_ylabel("candidate weight")
    entropy_ax = ax_f.twinx()
    gamma, entropy_values = theory_series(rows, "f", "normalized_entropy")
    entropy_ax.plot(gamma, entropy_values, color=COLORS["blue"], linestyle="--", lw=1.2)
    entropy_ax.set_ylim(0, 1.03)
    entropy_ax.set_ylabel(r"$H(w)/\log K$", color=COLORS["blue"], labelpad=2)
    entropy_ax.tick_params(axis="y", colors=COLORS["blue"], pad=1)
    entropy_ax.spines["right"].set_visible(True)
    entropy_ax.spines["right"].set_color(COLORS["blue"])

    add_panel_label(ax_g, "d")
    ax_g.set_title("Discounted evidence memory", loc="left", pad=4)
    rho, memory = theory_series(rows, "g", "effective_memory")
    point_rho, point_memory = theory_series(rows, "g", "formal_operating_point")
    ax_g.plot(rho, memory, color=COLORS["blue"], lw=1.7)
    ax_g.scatter(point_rho, point_memory, s=24, color=COLORS["orange"], edgecolor="white", linewidth=0.6, zorder=5)
    ax_g.axvline(point_rho[0], color=COLORS["orange"], linestyle="--", lw=0.8)
    ax_g.set_xlim(0, 1.0)
    ax_g.set_yscale("log")
    ax_g.set_ylim(1, 450)
    ax_g.set_xlabel(r"forgetting factor $\rho$")
    ax_g.set_ylabel(r"effective memory $N_{\rm eff}$")
    base = output_dir / "supp_shadow_theory"
    save_figure(fig, base)
    return base


def midpoint_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def organize_dynamics(rows: list[dict[str, str]]) -> dict[str, Any]:
    data: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for case in ("wave", "spring", "heat"):
        case_rows = [row for row in rows if row["case"] == case]
        for method in ("BMA", "PCE", "APCE"):
            entropy_rows = [
                row for row in case_rows if row["record_type"] == "entropy" and row["method"] == method
            ]
            entropy_rows.sort(key=lambda row: int(row["analysis_index"]))
            data[case][method]["entropy_time"] = np.asarray([float(row["time"]) for row in entropy_rows])
            data[case][method]["entropy"] = np.asarray(
                [float(row["normalized_entropy"]) for row in entropy_rows]
            )
            if method == "BMA":
                continue
            weight_rows = [
                row for row in case_rows if row["record_type"] == "weight" and row["method"] == method
            ]
            analyses = sorted({int(row["analysis_index"]) for row in weight_rows})
            candidates = sorted({int(row["candidate_index"]) for row in weight_rows})
            matrix = np.empty((len(candidates), len(analyses)), dtype=float)
            times = np.empty(len(analyses), dtype=float)
            alpha = np.empty(len(candidates), dtype=float)
            lookup = {
                (int(row["candidate_index"]), int(row["analysis_index"])): row for row in weight_rows
            }
            for col, analysis_index in enumerate(analyses):
                for row_index, candidate_index in enumerate(candidates):
                    row = lookup[(candidate_index, analysis_index)]
                    matrix[row_index, col] = float(row["weight"])
                    times[col] = float(row["time"])
                    alpha[row_index] = float(row["alpha"])
            data[case][method]["weights"] = matrix
            data[case][method]["time"] = times
            data[case][method]["alpha"] = alpha
            data[case][method]["alpha_true"] = float(weight_rows[0]["alpha_true"])
            data[case][method]["seed"] = int(weight_rows[0]["seed"])
    return data


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


def make_dynamics_figure(rows: list[dict[str, str]], output_dir: Path) -> Path:
    data = organize_dynamics(rows)
    fig = plt.figure(figsize=(7.2, 6.25))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        left=0.09,
        right=0.985,
        bottom=0.14,
        top=0.91,
        hspace=0.37,
        wspace=0.35,
    )
    cases = ("wave", "spring", "heat")
    case_titles = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
    panel_labels = iter("abcdefghi")
    heatmap_axes: list[plt.Axes] = []
    image = None

    for row_index, case in enumerate(cases):
        for col_index, method in enumerate(("PCE", "APCE")):
            ax = fig.add_subplot(grid[row_index, col_index])
            heatmap_axes.append(ax)
            method_data = data[case][method]
            x_edges = midpoint_edges(method_data["time"])
            y_edges = midpoint_edges(method_data["alpha"])
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                method_data["weights"],
                cmap=WEIGHT_CMAP,
                norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
                shading="flat",
                rasterized=False,
            )
            truth_line = ax.axhline(
                method_data["alpha_true"], color="white", linestyle="--", lw=1.8, zorder=4
            )
            truth_line.set_path_effects([path_effects.Stroke(linewidth=2.6, foreground=COLORS["ink"]), path_effects.Normal()])
            ax.set_xlim(float(method_data["time"].min()), float(method_data["time"].max()))
            ax.set_ylim(0.075, 0.40)
            ax.set_yticks([0.08, 0.12, 0.20, 0.30, 0.395])
            ax.set_yticklabels(["0.08", "0.12", "0.20", "0.30", "0.395"] if col_index == 0 else [])
            if row_index == 2:
                ax.set_xlabel("time")
            else:
                ax.set_xticklabels([])
            if col_index == 0:
                ax.set_ylabel(r"candidate $\alpha$")
            if row_index == 0:
                ax.set_title(f"{method} path weights", pad=5)
            add_panel_label(ax, next(panel_labels))
            if col_index == 0:
                ax.text(
                    -0.30,
                    0.50,
                    case_titles[case],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                )
            if row_index == 0 and col_index == 1:
                ax.text(
                    0.98,
                    0.95,
                    r"$\alpha^\star=0.12$",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.0,
                    color=COLORS["ink"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
                )

        entropy_ax = fig.add_subplot(grid[row_index, 2])
        for method in ("BMA", "PCE", "APCE"):
            entropy_ax.plot(
                data[case][method]["entropy_time"],
                data[case][method]["entropy"],
                color=METHOD_COLORS[method],
                lw=1.45,
                label=method,
            )
        entropy_ax.set_ylim(-0.02, 1.03)
        entropy_ax.set_xlim(0, max(data[case]["PCE"]["entropy_time"]))
        entropy_ax.set_yticks([0, 0.5, 1.0])
        if row_index == 2:
            entropy_ax.set_xlabel("time")
        else:
            entropy_ax.set_xticklabels([])
        entropy_ax.set_ylabel(r"$H(w)/\log K$")
        if row_index == 0:
            entropy_ax.set_title("Weight concentration", pad=5)
        add_panel_label(entropy_ax, next(panel_labels))

    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=1.6, label=m) for m in ("BMA", "PCE", "APCE")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=3, columnspacing=1.2, handlelength=2.0)
    if image is None:
        raise RuntimeError("No heatmap was generated")
    cbar_ax = fig.add_axes([0.24, 0.055, 0.52, 0.018])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("candidate weight", labelpad=1.5)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=6.3, pad=1.0, length=2.0)
    base = output_dir / "supp_cognitive_weight_dynamics"
    save_figure(fig, base)
    return base


def inspect_outputs(bases: list[Path]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for base in bases:
        figure_outputs: dict[str, Any] = {}
        for suffix in (".svg", ".pdf", ".png", ".tiff"):
            path = base.with_suffix(suffix)
            details: dict[str, Any] = {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            if suffix in {".png", ".tiff"}:
                with Image.open(path) as image:
                    details["pixels"] = list(image.size)
                    details["dpi"] = [float(value) for value in image.info.get("dpi", (0, 0))]
            if suffix == ".svg":
                svg_text = path.read_text(encoding="utf-8")
                details["editable_text_nodes"] = svg_text.count("<text")
                if details["editable_text_nodes"] == 0:
                    raise ValueError(f"{path} does not contain editable SVG text nodes")
            figure_outputs[suffix.lstrip(".")] = details
        outputs[base.name] = figure_outputs
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the Supplementary theory figures from exported CSV files.")
    parser.add_argument(
        "--source-dir", type=Path, default=ROOT / "ncs_english_latex" / "source_data"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "ncs_english_latex" / "figures-supplemental"
    )
    args = parser.parse_args()

    set_style()
    theory_path = args.source_dir / "supp_shadow_theory_source_data.csv"
    dynamics_path = args.source_dir / "supp_cognitive_weight_dynamics_source_data.csv"
    manifest_path = args.source_dir / "supp_theory_figures_manifest.json"
    theory_rows = read_csv(theory_path)
    dynamics_rows = read_csv(dynamics_path)

    theory_base = make_theory_figure(theory_rows, args.output_dir)
    dynamics_base = make_dynamics_figure(dynamics_rows, args.output_dir)
    outputs = inspect_outputs([theory_base, dynamics_base])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = outputs
    manifest.setdefault("software", {}).update(
        {"matplotlib": matplotlib.__version__, "pillow": Image.__version__, "platform": platform.platform()}
    )
    manifest["qa"] = {
        "backend_exclusive": "Python/matplotlib",
        "svg_text_editable": True,
        "raster_target_dpi": 600,
        "analytical_status_defined_in_caption": True,
        "trajectory_evidence_boundary_defined_in_caption": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
