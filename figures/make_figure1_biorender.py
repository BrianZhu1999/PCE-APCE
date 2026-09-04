"""Original BioRender-inspired vector schematic for manuscript Figure 1.

The illustration is intentionally schematic: it visualizes the method's
information flow and does not encode quantitative benchmark results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Ellipse, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
STEM = OUT / "figure1_framework_biorender_v1"

INK = "#1F2937"
MUTED = "#64748B"
GRID = "#D7DEE8"
BLUE = "#2F67A6"
BLUE_LT = "#E7F0FB"
ORANGE = "#D97922"
ORANGE_LT = "#FBEBD9"
VIOLET = "#6E5BAA"
VIOLET_LT = "#F0ECFA"
TEAL = "#2F7F7B"
TEAL_LT = "#E3F4F1"
RED = "#B84242"
GREEN = "#3C8C70"


def box(ax, x, y, w, h, face, edge=GRID, lw=1.2, radius=0.025, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.007,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, a, b, color=INK, lw=1.6, ms=12, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        linestyle=ls, color=color, shrinkA=0, shrinkB=0, zorder=z,
    ))


def text(ax, x, y, s, size=8, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, **kw)


def field(ax, x, y, w, h, sensors=True, phase=0.0):
    n = 100
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    z = (0.9 * np.sin(2.1 * np.pi * (yy + 0.12 * np.sin(2 * np.pi * xx) + phase))
         + 0.15 * np.cos(2.5 * np.pi * xx - 1.2 * yy))
    ax.imshow(z, extent=(x, x + w, y, y + h), origin="lower", cmap="RdYlBu_r",
              vmin=-1.05, vmax=1.05, interpolation="bilinear", zorder=2)
    # contour-like streamlines
    for off in np.linspace(0.14, 0.86, 5):
        t = np.linspace(0, 1, 120)
        yyline = y + h * (off + 0.08 * np.sin(2 * np.pi * (t + off + phase)))
        ax.plot(x + w * t, yyline, color="white", lw=0.6, alpha=0.7, zorder=3)
    if sensors:
        pts = np.array([[.15,.24],[.30,.69],[.48,.39],[.67,.78],[.83,.30],[.88,.62]])
        ax.scatter(x + w*pts[:,0], y + h*pts[:,1], s=42, facecolor="white",
                   edgecolor=INK, linewidth=0.8, zorder=4)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#AAB7C7", linewidth=.9, zorder=5))


def cloud(ax, x, y, color, scale=1.0):
    rng = np.random.default_rng(7 if color == BLUE else 11)
    pts = rng.normal([x, y], [0.018*scale, 0.010*scale], size=(120, 2))
    ax.scatter(pts[:,0], pts[:,1], s=5, color=color, alpha=.18, linewidths=0, zorder=4)
    ax.add_patch(Ellipse((x, y), .065*scale, .038*scale, facecolor=color,
                        edgecolor="white", linewidth=.8, alpha=.82, zorder=5))


def ensemble(ax, x0, y0, w, color, n=7):
    t = np.linspace(0, 1, 80)
    for i in range(n):
        y = y0 + (i-(n-1)/2)*.012 + .018*np.sin(2*np.pi*t + .4*i)
        ax.plot(x0+w*t, y, color=color, lw=1.0, alpha=.75, zorder=4)


def evidence(ax, x0, y0, w, color):
    t = np.linspace(0, 1, 90)
    for i, mu in enumerate([.18,.35,.52,.69]):
        y = y0 + .006*i + .035*np.exp(-((t-mu)/.07)**2)
        ax.plot(x0+w*t, y, color=color, lw=1.0, alpha=.8, zorder=4)


def branch(ax, x, y, w, h, color, face, title, subtitle, evidence_only=False):
    box(ax, x, y, w, h, face, color, lw=1.6, radius=.025)
    text(ax, x+w/2, y+h-.045, title, size=10.5, color=color, weight="bold")
    text(ax, x+w/2, y+h-.088, subtitle, size=7.6, color=INK)
    cloud(ax, x+.08, y+.045, color, 1.0)
    ensemble(ax, x+.14, y+.041, .12, color)
    if evidence_only:
        evidence(ax, x+.30, y+.035, .12, color)
    else:
        cloud(ax, x+.36, y+.045, color, .9)
    arrow(ax, (x+.115, y+.045), (x+.14, y+.045), color=color, lw=1.0, ms=8)
    arrow(ax, (x+.275, y+.045), (x+.30, y+.045), color=color, lw=1.0, ms=8)


def main():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    text(ax, .50, .955, "Training-free hybrid stochastic–cognitive assimilation", size=22, weight="bold")
    text(ax, .50, .918, "Shadow-anchored predictive evidence separates cognitive-path selection from state analysis", size=11.5, color=MUTED)

    # Panel headings
    text(ax, .045, .862, "a", size=13, weight="bold", ha="left")
    text(ax, .067, .862, "Sparse sensing\nmixes two uncertainties", size=9.3, weight="bold", ha="left", va="top", linespacing=1.05)
    text(ax, .345, .862, "b", size=13, weight="bold", ha="left")
    text(ax, .367, .862, "Shadow evidence vs.\nstate analysis", size=9.3, weight="bold", ha="left", va="top", linespacing=1.05)
    text(ax, .735, .862, "c", size=13, weight="bold", ha="left")
    text(ax, .757, .862, "Outputs and\nsensing-loss forecast", size=9.3, weight="bold", ha="left", va="top", linespacing=1.05)

    # Panel a: field and alpha paths
    box(ax, .035, .235, .265, .57, "#F9FBFD", GRID, lw=1.0, radius=.02)
    text(ax, .167, .770, "latent stochastic field", size=9.2, weight="bold")
    field(ax, .060, .330, .215, .360, sensors=True)
    text(ax, .167, .285, "sparse / partial observations", size=8, color=INK)
    text(ax, .167, .245, "within-path stochasticity", size=7.4, color=TEAL)
    text(ax, .167, .220, "between-path cognitive uncertainty", size=7.4, color=VIOLET)
    # candidate alpha paths
    box(ax, .055, .700, .090, .040, VIOLET_LT, VIOLET, lw=1.0, radius=.012)
    text(ax, .100, .753, r"candidate paths  α$_k$", size=7.0, color=VIOLET, weight="bold")
    for i, c in enumerate(["#9C8ED0", "#7A6AB7", "#5D4C9E"]):
        t = np.linspace(.07, .135, 60)
        ax.plot(t, .705 + .009*i + .008*np.sin(80*t+i), color=c, lw=1.1, zorder=4)
    arrow(ax, (.145, .720), (.177, .720), color=VIOLET, lw=1.2, ms=9)
    text(ax, .205, .720, r"q(α$_k$) → candidate equation", size=7.1, color=INK, ha="left")

    # Panel b: core architecture
    box(ax, .345, .185, .365, .620, "#FFFFFF", GRID, lw=1.0, radius=.02)
    box(ax, .360, .430, .085, .130, "#F8FAFC", GRID, lw=1.0, radius=.018)
    text(ax, .402, .535, "paired", size=8.2, weight="bold")
    text(ax, .402, .505, "initial", size=8.2, weight="bold")
    text(ax, .402, .475, "ensemble", size=8.2, weight="bold")
    ensemble(ax, .372, .455, .058, VIOLET, n=6)
    arrow(ax, (.445, .525), (.480, .665), color=BLUE, lw=1.7, ms=11)
    arrow(ax, (.445, .465), (.480, .315), color=ORANGE, lw=1.7, ms=11)
    branch(ax, .480, .585, .190, .160, BLUE, BLUE_LT, "Shadow branch", "forecast evidence only", True)
    branch(ax, .480, .255, .190, .160, ORANGE, ORANGE_LT, "Analysis branch", "observation-conditioned state", False)
    # observations enter only analysis branch
    box(ax, .490, .195, .170, .038, TEAL_LT, TEAL, lw=1.0, radius=.012)
    text(ax, .575, .214, r"observations  y$_t$ = H$_t$x$_t$ + ε$_t$", size=7.4, weight="bold")
    arrow(ax, (.575, .233), (.575, .255), color=ORANGE, lw=1.2, ms=9)
    # no feedback guardrail
    ax.plot([.482,.670],[.565,.565], color=RED, lw=1.5, ls=(0,(3,3)), zorder=4)
    text(ax, .575, .545, "no analysis feedback into evidence", size=7.4, color=RED, weight="bold")
    # score + weights + mixture
    box(ax, .545, .775, .145, .042, BLUE_LT, BLUE, lw=1.0, radius=.012)
    text(ax, .617, .796, "shadow predictive score", size=7.0, color=BLUE, weight="bold")
    arrow(ax, (.575, .585), (.575, .817), color=BLUE, lw=1.1, ms=8)
    # Compact hand-off labels keep the branch panel readable at journal width.
    text(ax, .646, .742, "evidence", size=7.4, color=BLUE, weight="bold", ha="left")
    arrow(ax, (.670, .665), (.704, .720), color=BLUE, lw=1.4, ms=10)
    arrow(ax, (.670, .335), (.704, .300), color=ORANGE, lw=1.4, ms=10)
    text(ax, .645, .455, "shared shadow anchor", size=7.0, color=BLUE, weight="bold")

    # Panel c: outputs
    box(ax, .730, .535, .235, .270, "#F9FBFD", GRID, lw=1.0, radius=.02)
    text(ax, .847, .770, "Weighted state mixture", size=9.5, weight="bold")
    text(ax, .847, .737, r"x̂$_t$ = Σ$_k$ w$_k$ x̂$_t^{(k)}$   ·   state + uncertainty", size=7.8, color=MUTED)
    field(ax, .752, .585, .052, .125, sensors=True, phase=.00)
    field(ax, .820, .585, .052, .125, sensors=False, phase=.12)
    field(ax, .888, .585, .052, .125, sensors=False, phase=.23)
    arrow(ax, (.806, .647), (.816, .647), color=MUTED, lw=1.2, ms=8)
    arrow(ax, (.874, .647), (.884, .647), color=MUTED, lw=1.2, ms=8)
    text(ax, .778, .560, "observed", size=7.2, color=MUTED)
    text(ax, .846, .560, "reference", size=7.2, color=MUTED)
    text(ax, .914, .560, "estimate", size=7.2, color=MUTED)
    box(ax, .730, .235, .235, .245, "#F9FBFD", GRID, lw=1.0, radius=.02)
    text(ax, .847, .450, "Forecast after sensing loss", size=10, weight="bold")
    ax.add_patch(Rectangle((.752,.285), .190, .105, facecolor="#EAF2FA", edgecolor="none", zorder=2))
    t = np.linspace(.752,.942,160)
    ax.plot(t, .345+.026*np.sin(55*t), color=BLUE, lw=2.1, zorder=4)
    ax.plot(t, .345+.026*np.sin(55*t+.8)+.008*np.sin(15*t), color=INK, lw=1.3, alpha=.75, zorder=4)
    ax.plot([.847,.847],[.282,.393], color=INK, lw=1.0, ls=(0,(3,2)), zorder=5)
    text(ax, .847, .400, "sensing loss", size=7.2, color=INK)
    text(ax, .847, .260, "freeze evidence path; propagate the state mixture", size=7.2, color=MUTED)

    # protocol strip
    box(ax, .035, .075, .930, .085, "#F5F8FC", GRID, lw=1.0, radius=.018)
    text(ax, .075, .132, "paired evaluation protocol", size=8.7, weight="bold", ha="left")
    labels = ["same truth trajectories", "same observations", "same initial ensembles", "same forecast perturbations", "paired seeds", "training-free"]
    xs = np.linspace(.235, .920, len(labels))
    for i, (x, lab) in enumerate(zip(xs, labels)):
        if i:
            arrow(ax, (xs[i-1]+.045, .118), (x-.045, .118), color=GRID, lw=1.0, ms=7)
        Circle((x, .118), .011, facecolor=[TEAL,TEAL,BLUE,BLUE,VIOLET,GREEN][i], edgecolor="white", linewidth=.8, zorder=4)
        ax.add_patch(Circle((x, .118), .011, facecolor=[TEAL,TEAL,BLUE,BLUE,VIOLET,GREEN][i], edgecolor="white", linewidth=.8, zorder=4))
        text(ax, x, .092, lab, size=6.8, color=INK)

    # legend
    y = .035
    for x, c, lab in [(.365, BLUE, "shadow evidence"), (.555, ORANGE, "state analysis"), (.750, VIOLET, "cognitive weights")]:
        ax.plot([x-.028,x+.002],[y,y], color=c, lw=3.0, solid_capstyle="round")
        text(ax, x+.040, y, lab, size=6.9, color=INK, ha="left")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
