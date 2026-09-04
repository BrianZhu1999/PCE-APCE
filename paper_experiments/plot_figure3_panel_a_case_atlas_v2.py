"""Stand-alone sketch for Figure 3 panel a.

This version prioritizes board-style composition over density:
one compact title band, one hero atlas board, and one clean evidence box.
It is intentionally closer to a schematic plate than to a statistical chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8


OUT = Path(
    r".\hybrid_uncertain_wave\ncs_chinese_submission\figures\figure3_panel_a_case_atlas_sketch_v2"
)
OUT.parent.mkdir(parents=True, exist_ok=True)


BG = "#FBF7EE"
BOARD = "#F7F0E2"
BOARD_EDGE = "#D7C19A"
ORANGE = "#D58A30"
TEAL = "#4A9BAE"
TEXT = "#222222"
SUBTLE = "#606060"


def draw_curve_icon(ax, x, y, w, h, kind, color_main, color_aux):
    t = np.linspace(0, 1, 120)
    if kind == "chemical":
        ax.plot(x + 0.02 + 0.11 * t, y + 0.020 * np.sin(4.0 * t) + 0.008 * (1 - t), color=color_aux, lw=1.1)
        ax.plot(x + 0.02 + 0.11 * t, y - 0.006 + 0.016 * np.sin(2.3 * t + 0.4), color=color_main, lw=1.2)
    elif kind == "pk":
        ax.plot(x + 0.03 + 0.10 * t, y + 0.020 * np.exp(-4.5 * t), color=color_main, lw=1.25)
        ax.plot(x + 0.03 + 0.10 * t, y + 0.003 * np.exp(-1.7 * t), color=color_aux, lw=1.05)
    elif kind == "sir":
        ax.plot(x + 0.02 + 0.11 * t, y + 0.017 * np.exp(-4.0 * t), color=color_aux, lw=1.0)
        ax.plot(x + 0.02 + 0.11 * t, y + 0.009 + 0.010 * np.sin(4.0 * t), color=color_main, lw=1.15)
        ax.plot(x + 0.02 + 0.11 * t, y - 0.004 + 0.014 / (1 + np.exp(-8 * (t - 0.5))), color="#A1A1A1", lw=1.0)
    elif kind == "pendulum":
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(x + 0.055 + 0.042 * np.cos(th), y + 0.008 + 0.020 * np.sin(th), color=color_main, lw=1.15)
        ax.plot([x + 0.055, x + 0.062], [y + 0.033, y + 0.006], color=color_aux, lw=1.0)
        ax.scatter([x + 0.062], [y + 0.006], s=9, color=color_main, zorder=4)
    elif kind == "fhn":
        ax.plot(x + 0.02 + 0.11 * t, y + 0.004 + 0.019 * np.sin(1.25 * np.pi * t), color=color_main, lw=1.15)
        ax.plot(x + 0.02 + 0.11 * t, y + 0.010 * np.cos(2 * np.pi * t), color=color_aux, lw=1.0)
    elif kind == "robertson":
        ax.plot(x + 0.02 + 0.11 * t, y + 0.024 * np.exp(-3.0 * t), color=color_aux, lw=1.0)
        ax.plot(x + 0.02 + 0.11 * t, y - 0.004 + 0.013 * np.sin(4.0 * t), color=color_main, lw=1.15)
        ax.plot(x + 0.02 + 0.11 * t, y + 0.004 + 0.007 * np.cos(5.0 * t), color="#A1A1A1", lw=1.0)


fig = plt.figure(figsize=(4.25, 6.45), dpi=240)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_axis_off()
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# panel label and title
ax.text(0.03, 0.975, "a", fontsize=16, fontweight="bold", ha="left", va="top", color="#111111")
ax.text(0.11, 0.975, "Applied ODE case atlas", fontsize=14, ha="left", va="top", color="#111111")

# compact strapline
strap = FancyBboxPatch(
    (0.085, 0.885),
    0.83,
    0.058,
    boxstyle="round,pad=0.010,rounding_size=0.018",
    facecolor="#FFFDF8",
    edgecolor=BOARD_EDGE,
    linewidth=0.8,
)
ax.add_patch(strap)
ax.text(
    0.50,
    0.914,
    "selected-six formal atlas  ·  source-derived + canonical stress tests  ·  50 paired seeds",
    fontsize=8.0,
    ha="center",
    va="center",
    color="#535353",
)

# large hero board
hero = FancyBboxPatch(
    (0.065, 0.30),
    0.87,
    0.53,
    boxstyle="round,pad=0.014,rounding_size=0.024",
    facecolor=BOARD,
    edgecolor=BOARD_EDGE,
    linewidth=1.0,
)
ax.add_patch(hero)

# section headers
ax.text(0.10, 0.795, "Source-derived uncertain ODEs", fontsize=9.4, ha="left", va="center", color="#8B5B21")
ax.plot([0.10, 0.90], [0.775, 0.775], color=ORANGE, lw=0.9, alpha=0.8)
ax.text(0.10, 0.520, "Canonical stress tests", fontsize=9.4, ha="left", va="center", color="#346C79")
ax.plot([0.10, 0.90], [0.500, 0.500], color=TEAL, lw=0.9, alpha=0.8)


def card(x, y, w, h, title, edge, face, kind, badge=None):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.016",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.add_line(plt.Line2D([x + 0.01, x + w - 0.01], [y + h - 0.013, y + h - 0.013], color=edge, lw=0.65, alpha=0.6))
    if badge:
        ax.text(x + 0.017, y + h - 0.031, badge, fontsize=6.4, color=edge, ha="left", va="center")
    draw_curve_icon(ax, x, y + 0.025, w, h, kind, edge, "#666666")
    ax.text(x + w / 2, y + 0.012, title, fontsize=9.0, ha="center", va="bottom", color=TEXT)


top_y = 0.60
bot_y = 0.375
card_w = 0.245
card_h = 0.145
gap = 0.035
x_positions = [0.10, 0.10 + card_w + gap, 0.10 + 2 * (card_w + gap)]

for x, title, kind in zip(x_positions, ["Chemical", "PK", "SIR"], ["chemical", "pk", "sir"]):
    card(x, top_y, card_w, card_h, title, ORANGE, "#FFF8F0", kind, badge="source")

for x, title, kind in zip(x_positions, ["Pendulum", "FHN", "Robertson"], ["pendulum", "fhn", "robertson"]):
    card(x, bot_y, card_w, card_h, title, TEAL, "#F2FBFD", kind, badge="stress")

# bottom evidence box
box = FancyBboxPatch(
    (0.17, 0.095),
    0.66,
    0.125,
    boxstyle="round,pad=0.012,rounding_size=0.016",
    facecolor="#FFFDF9",
    edgecolor="#2E2E2E",
    linewidth=0.9,
)
ax.add_patch(box)
ax.text(0.50, 0.160, "PCE / APCE shared core", fontsize=10.0, ha="center", va="center", color="#151515")
ax.text(0.50, 0.130, "evidence-weighted shadow forecast", fontsize=8.0, ha="center", va="center", color="#444444")
ax.text(0.50, 0.107, "state error  ·  interval width  ·  cognitive-coordinate error", fontsize=7.1, ha="center", va="center", color="#666666")

ax.text(0.50, 0.035, "provenance-first atlas, not a trajectory collage", fontsize=7.3, ha="center", va="center", color="#4C4C4C")

fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
plt.close(fig)

print(str(OUT))
