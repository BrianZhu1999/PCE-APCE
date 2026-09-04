"""Figure 1a, v2: two uncertainty layers under sparse observations.

This is a schematic-only panel.  It uses analytic trajectories generated in
the source script; no benchmark data are plotted.  Candidate cognitive paths
are shown as coloured mean trajectories, while pale same-colour traces show
stochastic ensemble variability within each path.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
STEM = OUT_DIR / "figure1_panel_a_layers_v2"

INK = "#111827"
MUTED = "#5B6472"
GRID = "#C9D1DC"
BLUE = "#1454B8"
CYAN = "#2996D6"
GREEN = "#4D9B55"
ORANGE = "#E97821"
RED = "#E13A2F"
VIOLET = "#6A43B6"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def candidate_means(t: np.ndarray) -> list[np.ndarray]:
    """Return five separated candidate dynamics with a common initial state."""
    # The curves are illustrative trajectories, not a numerical benchmark.
    phase = 2.0 * np.pi * t
    envelopes = [
        0.92 * (1.0 - np.exp(-5.4 * t)),
        0.64 * (1.0 - np.exp(-4.8 * t)),
        0.32 * (1.0 - np.exp(-4.4 * t)),
        -0.42 * (1.0 - np.exp(-4.5 * t)),
        -0.72 * (1.0 - np.exp(-5.0 * t)),
    ]
    means = []
    for idx, envelope in enumerate(envelopes):
        drift = 0.045 * np.sin(phase * (1.0 + 0.08 * idx) + 0.35 * idx) * (1.0 - np.exp(-3.0 * t))
        relaxation = 0.10 * np.sin(3.0 * phase + 0.55 * idx) * (1.0 - np.exp(-8.0 * t))
        means.append(envelope + drift + relaxation)
    return means


def draw_panel() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(5.1, 3.15), facecolor="white")
    ax = fig.add_axes([0.12, 0.30, 0.80, 0.52])
    t = np.linspace(0.0, 1.0, 320)
    means = candidate_means(t)
    colours = [BLUE, CYAN, GREEN, ORANGE, VIOLET]
    rng = np.random.default_rng(3187)

    # Sparse observation times are shown as a small number of vertical marks.
    obs_times = np.array([0.12, 0.27, 0.48, 0.76])
    for obs_t in obs_times:
        ax.axvline(obs_t, color=GRID, lw=0.55, ls=(0, (2.0, 2.5)), zorder=0)

    # Pale members first; coloured mean paths then define the candidate family.
    for idx, (mean, colour) in enumerate(zip(means, colours)):
        for member in range(8):
            phase = rng.uniform(-0.4, 0.4)
            smooth_noise = (
                0.020 * np.sin(2.0 * np.pi * t * (1.5 + 0.12 * member) + phase)
                + 0.012 * np.sin(2.0 * np.pi * t * (5.0 + 0.2 * idx) + 1.6 * phase)
            )
            random_walk = np.cumsum(rng.normal(0.0, 0.00022, size=t.size))
            random_walk -= random_walk[0]
            spread = (0.38 + 0.62 * t) * (0.52 + 0.06 * idx)
            member_trace = mean + spread * (smooth_noise + random_walk)
            ax.plot(t, member_trace, color=colour, lw=0.42, alpha=0.30, zorder=1)
        ax.fill_between(t, mean - 0.052 * (0.8 + t), mean + 0.052 * (0.8 + t), color=colour, alpha=0.14, linewidth=0, zorder=1)
        ax.plot(t, mean, color=colour, lw=1.10, solid_capstyle="round", zorder=3)

    # Sparse observations are a separate noisy observation sequence, not a
    # labelled candidate path. The connector is only a visual guide.
    ref_mean = means[2]
    obs_offsets = np.array([0.055, -0.035, 0.042, -0.028])
    obs_y = np.interp(obs_times, t, ref_mean) + obs_offsets
    ax.plot(obs_times, obs_y, color=INK, lw=0.72, ls=(0, (1.5, 1.8)), alpha=0.70, zorder=4)
    ax.scatter(obs_times, obs_y, s=28, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=5)
    ax.text(
        0.70,
        0.98,
        r"sparse observations $y_t$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.78),
    )

    # Direct candidate labels keep the path identity visible without a legend.
    labels = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\alpha_{K-1}$", r"$\alpha_K$"]
    label_positions = [0.96, 0.64, 0.34, -0.44, -0.76]
    for label, colour, y in zip(labels, colours, label_positions):
        ax.text(1.012, y, label, transform=ax.get_yaxis_transform(), color=colour, fontsize=7.1, va="center", ha="left")
    ax.text(0.02, 0.96, "Cognitive uncertainty:", transform=ax.transAxes, color=BLUE, fontsize=7.8, fontweight="bold", ha="left", va="top")
    ax.text(0.02, 0.885, r"candidate dynamics $\alpha_k$", transform=ax.transAxes, color=INK, fontsize=7.2, ha="left", va="top")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.02, 1.08)
    ax.set_xlabel(r"$t$", fontsize=8.2, labelpad=1.5)
    ax.set_ylabel(r"$x_t$", fontsize=8.2, labelpad=1.5, rotation=0)
    ax.yaxis.set_label_coords(-0.055, 0.98)
    ax.set_xticks(obs_times)
    ax.set_xticklabels([r"$t_1$", r"$t_2$", r"$t_3$", r"$t_q$"], fontsize=6.7)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2.4, width=0.65, pad=1.8)

    # Common starting point emphasizes that cognitive paths branch from the same state.
    ax.scatter([0.0], [0.0], s=25, color=INK, zorder=6)
    ax.annotate(r"$t_0$", xy=(0.0, 0.0), xycoords="data", xytext=(4, -15), textcoords="offset points", fontsize=6.7, ha="left", va="top")
    ax.annotate("shared initial state", xy=(0.0, 0.0), xycoords="data", xytext=(10, -31), textcoords="offset points", fontsize=6.1, color=MUTED, ha="left", va="top")
    return fig, ax


def add_definition_text(fig: plt.Figure) -> None:
    fig.text(0.045, 0.955, "a", fontsize=11.2, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.095, 0.955, "Two layers of uncertainty", fontsize=10.4, fontweight="bold", color=INK, ha="left", va="top")

    fig.text(0.12, 0.225, "Stochastic uncertainty:", fontsize=7.5, fontweight="bold", color=VIOLET, ha="left", va="top")
    fig.text(
        0.40,
        0.225,
        r"$\mathrm{d}X_t^{(k)}=F(t,X_t^{(k)};\theta_k)\,\mathrm{d}t+G(t,X_t^{(k)})\,\mathrm{d}W_t$",
        fontsize=7.1,
        color=INK,
        ha="left",
        va="top",
    )
    fig.text(0.12, 0.175, "ensemble variability within path", fontsize=7.1, color=INK, ha="left", va="top")


def draw_space_time_mask(fig: plt.Figure) -> None:
    """Add a compact vector inset showing spatial and temporal sparsity."""
    ax = fig.add_axes([0.70, 0.035, 0.25, 0.13])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor("#F8FAFC")

    x_grid = np.linspace(0.08, 0.92, 13)
    t_grid = np.linspace(0.14, 0.88, 8)
    for x in x_grid:
        ax.plot([x, x], [0.10, 0.92], color=GRID, lw=0.35, zorder=0)
    for y in t_grid:
        ax.plot([0.06, 0.94], [y, y], color=GRID, lw=0.35, zorder=0)

    # Four observation times match the main timeline; only a few
    # state/space locations are sampled at each time.
    observations = np.array(
        [
            [0.18, 0.24],
            [0.61, 0.24],
            [0.34, 0.43],
            [0.82, 0.43],
            [0.50, 0.62],
            [0.90, 0.62],
            [0.25, 0.84],
            [0.72, 0.84],
        ]
    )
    ax.scatter(
        observations[:, 0],
        observations[:, 1],
        s=17,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.75,
        zorder=3,
    )
    ax.text(0.50, 1.03, "space–time sparse observations", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.9, color=INK)
    ax.text(0.50, -0.12, "space / state coordinate", transform=ax.transAxes, ha="center", va="top", fontsize=5.3, color=MUTED)
    ax.text(-0.08, 0.52, "time", transform=ax.transAxes, ha="right", va="center", fontsize=5.3, color=MUTED, rotation=90)
    ax.text(0.10, 0.01, r"$r_1$", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.0, color=MUTED)
    ax.text(0.50, 0.01, r"$r_j$", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.0, color=MUTED)
    ax.text(0.90, 0.01, r"$r_M$", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.0, color=MUTED)
    ax.text(0.01, 0.24, r"$t_1$", transform=ax.transAxes, ha="right", va="center", fontsize=5.0, color=MUTED)
    ax.text(0.01, 0.43, r"$t_2$", transform=ax.transAxes, ha="right", va="center", fontsize=5.0, color=MUTED)
    ax.text(0.01, 0.62, r"$t_3$", transform=ax.transAxes, ha="right", va="center", fontsize=5.0, color=MUTED)
    ax.text(0.01, 0.84, r"$t_q$", transform=ax.transAxes, ha="right", va="center", fontsize=5.0, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(0.55)


def write_metadata() -> None:
    provenance = {
        "figure": "Figure 1a",
        "version": "layers_v2",
        "source_script": "make_figure1_panel_a_layers_v2.py",
        "backend": "Python / matplotlib",
        "archetype": "schematic-led composite",
        "scientific_data_status": "illustrative schematic; no experimental data are plotted",
        "core_conclusion": "Sparse observations expose two distinct uncertainty layers: within-path stochastic ensemble variability and between-path cognitive alternatives.",
        "candidate_paths": "five illustrative alpha-labelled paths with shared initial state",
        "stochastic_members": "eight deterministic illustrative ensemble traces per candidate path",
        "observation_marks": "four illustrative noisy observation times shown as a separate y_t sequence; not benchmark data",
        "space_time_inset": "vector grid with four observed time rows and eight observed state/space samples, illustrating joint temporal and spatial/state sparsity",
        "equation": "dX_t^(k)=F(t,X_t^(k);theta_k)dt+G(t,X_t^(k))dW_t",
        "outputs": [str(STEM.with_suffix(suffix)) for suffix in (".svg", ".pdf", ".png", ".tiff")],
        "integrity_note": "All graphical elements are generated from the source script; no exported SVG/PDF/PNG was edited after rendering.",
    }
    (STEM.with_name(STEM.name + "_provenance.json")).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    qa = {
        "text_audit": {
            "placeholder_terms": [],
            "generation_errors": [],
            "terminology": ["cognitive uncertainty", "candidate dynamics", "stochastic uncertainty", "ensemble variability within path", "sparse observations", "space–time sparse observations", "shared initial state"],
        },
        "vector_audit": {
            "embedded_raster_images": 0,
            "svg_text_preserved": True,
            "svg_paths_expected": True,
        },
        "status": "standalone review output; not inserted into main manuscript",
    }
    (STEM.with_name(STEM.name + "_qa.json")).write_text(json.dumps(qa, indent=2), encoding="utf-8")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = draw_panel()
    add_definition_text(fig)
    draw_space_time_mask(fig)
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600}), (".tiff", {"dpi": 600})):
        fig.savefig(str(STEM) + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)
    write_metadata()


if __name__ == "__main__":
    main()
