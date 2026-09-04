"""Figure 1 schematic: shadow-anchored evidence for hybrid assimilation.

This is a source-first redraw of the supplied conceptual reference.  It is a
schematic, not a quantitative result panel: the field plates are illustrative
and are generated deterministically from smooth functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
STEM = OUT_DIR / "figure1_framework_v2_shadow_anchor"


COLORS = {
    "ink": "#1F2937",
    "muted": "#667085",
    "blue": "#245A9A",
    "blue_light": "#DCEAF8",
    "orange": "#D97922",
    "orange_light": "#FCE7D5",
    "teal": "#2F7C7A",
    "teal_light": "#DDF2EE",
    "violet": "#6D5EA8",
    "violet_light": "#EEEAF9",
    "red": "#B33B3B",
    "line": "#C9D1DC",
    "panel": "#FAFBFC",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def rounded_box(ax: plt.Axes, xy, width, height, *, face, edge, radius=0.025, lw=1.0, alpha=1.0):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: plt.Axes, start, end, *, color=COLORS["ink"], lw=1.2, ms=11, style="-|>", z=5, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def field_data(n=120):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xx, yy = np.meshgrid(x, y)
    # A smooth wave-like field is only a visual placeholder for a latent field.
    z = (
        0.88 * np.sin(2.1 * np.pi * (yy + 0.12 * np.sin(2 * np.pi * xx)))
        + 0.18 * np.cos(2.7 * np.pi * xx - 1.4 * yy)
        + 0.08 * np.sin(5 * np.pi * xx + 3 * yy)
    )
    return xx, yy, z


def draw_field(ax: plt.Axes, *, show_sensors=False, phase=0.0, title=None):
    xx, yy, z = field_data()
    z = np.roll(z, int(phase * z.shape[1]), axis=1)
    ax.contourf(xx, yy, z, levels=24, cmap="RdYlBu_r", alpha=0.95)
    ax.contour(xx, yy, z, levels=9, colors="#FFFFFF", linewidths=0.25, alpha=0.5)
    # a few flowing paths to evoke state trajectories without implying data
    for offset in np.linspace(0.16, 0.84, 5):
        t = np.linspace(0.03, 0.97, 160)
        yy_path = offset + 0.08 * np.sin(2 * np.pi * (t + offset + phase))
        ax.plot(t, yy_path, color="#FFFFFF", lw=0.55, alpha=0.7)
    if show_sensors:
        sensor_xy = np.array(
            [
                [0.14, 0.23],
                [0.31, 0.70],
                [0.49, 0.39],
                [0.67, 0.78],
                [0.82, 0.31],
                [0.88, 0.63],
            ]
        )
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=24, facecolor="white", edgecolor=COLORS["ink"], linewidth=0.5, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.text(0.03, 1.02, title, transform=ax.transAxes, ha="left", va="bottom", color=COLORS["ink"], fontsize=7.5, fontweight="bold")


def tiny_ensemble(ax, color, *, n=9, y0=0.5, spread=0.22):
    x = np.linspace(0.05, 0.95, 90)
    for i in range(n):
        y = y0 + (i - (n - 1) / 2) * spread / max(n - 1, 1) + 0.035 * np.sin(2 * np.pi * x + i * 0.55)
        ax.plot(x, y, color=color, lw=0.75, alpha=0.75 if i not in (0, n - 1) else 0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.14, 0.86)
    ax.axis("off")


def density_cloud(ax, color, *, center=(0.5, 0.5), n=220, seed=3):
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center, scale=(0.12, 0.07), size=(n, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=2.5, alpha=0.18, color=color, edgecolors="none")
    ax.scatter([center[0]], [center[1]], s=25, color=color, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def mini_evidence(ax, color):
    x = np.linspace(0.06, 0.94, 120)
    for i, amp in enumerate([0.2, 0.35, 0.5, 0.28]):
        y = 0.2 + 0.12 * i + amp * np.exp(-((x - (0.26 + 0.16 * i)) / 0.09) ** 2)
        ax.plot(x, y, color=color, lw=0.8, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.9)
    ax.axis("off")


def draw_panel_a(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 1.02, "a", fontsize=10, fontweight="bold", va="bottom", color=COLORS["ink"])
    ax.text(0.07, 1.02, "Why sparse sensing is hard", fontsize=9.5, fontweight="bold", va="bottom", color=COLORS["ink"])

    inset = ax.inset_axes([0.03, 0.16, 0.88, 0.68])
    draw_field(inset, show_sensors=True)
    inset.text(0.03, 1.03, "latent stochastic field", transform=inset.transAxes, fontsize=7.2, color=COLORS["ink"], fontweight="bold")

    # within-path and between-path callouts
    ax.annotate("within-path\nstochasticity", xy=(0.42, 0.67), xycoords="axes fraction", xytext=(0.04, 0.82), textcoords="axes fraction", fontsize=7.0, color=COLORS["muted"], ha="left", va="center", arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.75))
    ax.annotate("between-path\ncognitive uncertainty", xy=(0.50, 0.28), xycoords="axes fraction", xytext=(0.02, 0.05), textcoords="axes fraction", fontsize=7.0, color=COLORS["violet"], ha="left", va="center", arrowprops=dict(arrowstyle="-", color=COLORS["violet"], lw=0.75))
    ax.text(0.72, 0.10, "sparse observations", fontsize=7.0, color=COLORS["ink"], ha="center")


def draw_branch_box(ax, *, x, y, width, title, subtitle, color, light, branch="shadow"):
    rounded_box(ax, (x, y), width, 0.27, face=light, edge=color, radius=0.02, lw=1.0)
    ax.text(x + width / 2, y + 0.224, title, ha="center", va="center", color=color, fontsize=8.0, fontweight="bold")
    ax.text(x + width / 2, y + 0.182, subtitle, ha="center", va="center", color=COLORS["ink"], fontsize=6.6)
    # inner mini stages
    mini1 = ax.inset_axes([x + 0.025, y + 0.055, 0.090, 0.092], transform=ax.transAxes)
    density_cloud(mini1, color, center=(0.5, 0.52), seed=5 if branch == "shadow" else 7)
    mini2 = ax.inset_axes([x + 0.145, y + 0.055, 0.090, 0.092], transform=ax.transAxes)
    tiny_ensemble(mini2, color, n=7, y0=0.5)
    mini3 = ax.inset_axes([x + 0.265, y + 0.055, 0.090, 0.092], transform=ax.transAxes)
    if branch == "shadow":
        mini_evidence(mini3, color)
    else:
        density_cloud(mini3, color, center=(0.57, 0.5), seed=11)
    arrow(ax, (x + 0.118, y + 0.10), (x + 0.145, y + 0.10), color=color, lw=0.8, ms=8)
    arrow(ax, (x + 0.238, y + 0.10), (x + 0.265, y + 0.10), color=color, lw=0.8, ms=8)


def draw_panel_b(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 1.02, "b", fontsize=10, fontweight="bold", va="bottom", color=COLORS["ink"])
    ax.text(0.07, 1.02, "Separate evidence from state analysis", fontsize=9.5, fontweight="bold", va="bottom", color=COLORS["ink"])

    # source / candidate ensemble on the left
    rounded_box(ax, (0.00, 0.35), 0.115, 0.28, face=COLORS["panel"], edge=COLORS["line"], radius=0.02, lw=0.8)
    ax.text(0.0575, 0.59, "candidate\npaths", ha="center", va="center", fontsize=5.8, fontweight="bold")
    ens_ax = ax.inset_axes([0.012, 0.40, 0.090, 0.12], transform=ax.transAxes)
    tiny_ensemble(ens_ax, COLORS["violet"], n=10, y0=0.5, spread=0.30)
    ax.text(0.0575, 0.375, "initial ensemble", ha="center", va="center", fontsize=6.0, color=COLORS["muted"])

    # branches
    draw_branch_box(ax, x=0.135, y=0.62, width=0.38, title="Shadow branch", subtitle="evidence only", color=COLORS["blue"], light=COLORS["blue_light"], branch="shadow")
    draw_branch_box(ax, x=0.135, y=0.16, width=0.38, title="Analysis branch", subtitle="state update", color=COLORS["orange"], light=COLORS["orange_light"], branch="analysis")
    arrow(ax, (0.115, 0.52), (0.135, 0.73), color=COLORS["blue"], lw=1.3, ms=10)
    arrow(ax, (0.115, 0.46), (0.135, 0.28), color=COLORS["orange"], lw=1.3, ms=10)

    # observation input and explicit separation
    obs = (0.19, 0.02, 0.24, 0.10)
    rounded_box(ax, obs[:2], obs[2], obs[3], face=COLORS["teal_light"], edge=COLORS["teal"], radius=0.018, lw=0.9)
    ax.text(0.31, 0.07, r"observations  $y_t=H_tX_t+\varepsilon_t$", ha="center", va="center", fontsize=6.4, color=COLORS["ink"])
    arrow(ax, (0.31, 0.12), (0.31, 0.16), color=COLORS["orange"], lw=1.0, ms=9)
    ax.plot([0.19, 0.46], [0.60, 0.60], color=COLORS["red"], lw=1.1, ls=(0, (2, 2)))
    ax.text(0.325, 0.575, "no feedback into evidence", ha="center", va="top", fontsize=5.9, color=COLORS["red"])

    # evidence and state mixture to the right
    rounded_box(ax, (0.55, 0.49), 0.17, 0.20, face=COLORS["violet_light"], edge=COLORS["violet"], radius=0.02, lw=0.95)
    ax.text(0.635, 0.645, "path weights", ha="center", va="center", fontsize=6.1, fontweight="bold", color=COLORS["violet"])
    ax.text(0.635, 0.58, r"$w_k,\ \hat{\alpha}_k$", ha="center", va="center", fontsize=8.0, color=COLORS["ink"])
    arrow(ax, (0.515, 0.755), (0.55, 0.62), color=COLORS["blue"], lw=1.2, ms=10)
    arrow(ax, (0.515, 0.295), (0.55, 0.55), color=COLORS["orange"], lw=1.2, ms=10)
    ax.text(0.53, 0.785, "predictive evidence", ha="center", va="center", fontsize=5.8, color=COLORS["blue"])
    ax.text(0.53, 0.255, "corrected state", ha="center", va="center", fontsize=5.8, color=COLORS["orange"])

    rounded_box(ax, (0.76, 0.46), 0.22, 0.26, face=COLORS["panel"], edge=COLORS["ink"], radius=0.02, lw=1.0)
    ax.text(0.87, 0.64, "state mixture", ha="center", va="center", fontsize=6.8, fontweight="bold")
    ax.text(0.87, 0.55, r"$\hat X_t=\sum_k w_k\hat X_t^{(k)}$", ha="center", va="center", fontsize=7.2)
    ax.text(0.87, 0.49, r"$\hat X_t\ \pm\ \sigma_t$", ha="center", va="center", fontsize=6.8, color=COLORS["muted"])
    arrow(ax, (0.72, 0.59), (0.76, 0.59), color=COLORS["violet"], lw=1.3, ms=10)
    ax.text(0.635, 0.36, "shared shadow anchor", ha="center", va="center", fontsize=6.0, color=COLORS["blue"], bbox=dict(boxstyle="round,pad=0.15", facecolor=COLORS["blue_light"], edgecolor=COLORS["blue"], linewidth=0.6))
    ax.text(0.87, 0.40, "state + uncertainty", ha="center", va="center", fontsize=5.9, color=COLORS["muted"])


def draw_panel_c(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.00, 1.02, "c", fontsize=10, fontweight="bold", va="bottom", color=COLORS["ink"])
    ax.text(0.07, 1.02, "What the separation enables", fontsize=9.5, fontweight="bold", va="bottom", color=COLORS["ink"])

    # reconstruction card
    rounded_box(ax, (0.02, 0.55), 0.96, 0.32, face=COLORS["panel"], edge=COLORS["line"], radius=0.025, lw=0.8)
    ax.text(0.50, 0.82, "Sparse-field reconstruction", ha="center", va="center", fontsize=7.5, fontweight="bold")
    for i, (x, title, phase) in enumerate([(0.10, "observations", 0.0), (0.39, "reference", 0.02), (0.68, "estimate", 0.04)]):
        iax = ax.inset_axes([x, 0.60, 0.20, 0.16], transform=ax.transAxes)
        draw_field(iax, show_sensors=(i == 0), phase=phase)
        iax.text(0.5, -0.14, title, transform=iax.transAxes, ha="center", va="top", fontsize=6.3, color=COLORS["muted"])
        if i < 2:
            arrow(ax, (x + 0.205, 0.68), (x + 0.26, 0.68), color=COLORS["muted"], lw=0.9, ms=8)

    # blackout card
    rounded_box(ax, (0.02, 0.10), 0.96, 0.32, face=COLORS["panel"], edge=COLORS["line"], radius=0.025, lw=0.8)
    ax.text(0.50, 0.37, "Forecast after sensing loss", ha="center", va="center", fontsize=7.5, fontweight="bold")
    iax = ax.inset_axes([0.08, 0.16, 0.76, 0.14], transform=ax.transAxes)
    x = np.linspace(0, 1, 240)
    base = 0.44 + 0.13 * np.sin(2 * np.pi * x)
    pred = base + 0.04 * np.sin(6 * np.pi * x + 0.4)
    iax.plot(x, base, color=COLORS["muted"], lw=1.2, label="observed trajectory")
    iax.plot(x, pred, color=COLORS["blue"], lw=1.5, label="forecast")
    iax.axvspan(0.45, 1.0, color=COLORS["blue_light"], alpha=0.65, lw=0)
    iax.axvline(0.45, color=COLORS["ink"], lw=0.8, ls=(0, (2, 2)))
    iax.text(0.45, 0.98, "sensing loss", transform=iax.transAxes, ha="center", va="top", fontsize=6.0, color=COLORS["ink"])
    iax.set_xlim(0, 1)
    iax.set_ylim(0.22, 0.70)
    iax.set_xticks([])
    iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_visible(False)
    ax.text(0.50, 0.125, "freeze the evidence path; propagate the state mixture", ha="center", va="center", fontsize=6.4, color=COLORS["muted"])


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.95), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.98, 1.80, 1.08], wspace=0.16, left=0.035, right=0.985, top=0.86, bottom=0.07)
    fig.text(0.5, 0.965, "Training-free hybrid stochastic–cognitive assimilation", ha="center", va="top", fontsize=13.5, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.918, "Shadow-anchored evidence separates cognitive-path selection from state analysis", ha="center", va="top", fontsize=7.9, color=COLORS["muted"])
    draw_panel_a(fig, gs[0])
    draw_panel_b(fig, gs[1])
    draw_panel_c(fig, gs[2])

    # a compact visual key rather than a repeated legend
    handles = [
        Line2D([0], [0], color=COLORS["blue"], lw=2, label="shadow evidence"),
        Line2D([0], [0], color=COLORS["orange"], lw=2, label="state analysis"),
        Line2D([0], [0], color=COLORS["violet"], lw=2, label="cognitive weights"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.005), ncol=3, fontsize=6.8, handlelength=1.7, columnspacing=1.4, frameon=False)

    stem = str(STEM)
    fig.savefig(stem + ".svg", bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(stem + ".tiff", dpi=600, bbox_inches="tight")
    qa = {
        "figure": "Figure 1",
        "version": "v2_shadow_anchor",
        "backend": "python/matplotlib",
        "archetype": "schematic-led composite",
        "core_conclusion": "Shadow-anchored evidence separates cognitive-path selection from state analysis in training-free hybrid assimilation.",
        "panels": {
            "a": "sparse sensing and hybrid uncertainty",
            "b": "shadow/analysis separation and weighted state mixture",
            "c": "conceptual reconstruction and sensing-loss forecast capabilities",
        },
        "scientific_data": "No experimental data; field plates and trajectories are deterministic schematic illustrations.",
        "outputs": [str(STEM.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
    }
    (STEM.with_name(STEM.name + "_qa.json")).write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
