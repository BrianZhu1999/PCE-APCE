from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
L63_84_SOURCE = ROOT / "results_lorenz63_84_gate_5seeds" / "representative_source"
L96_SOURCE = ROOT / "results_lorenz_ks_gate_5seeds_l96obs20" / "representative_source"
OUTPUT = ROOT / "figures" / "lorenz_family_trajectory_grid"

PALETTE = {
    "truth": "#E8898F",
    "denkf": "#7D8CB4",
    "letkf": "#A9B4CC",
    "pce": "#4F7FC1",
    "apce": "#164F91",
    "text": "#242424",
}
METHODS: tuple[str | None, ...] = (None, "denkf", "letkf", "pce", "apce")
TITLES = ("GT", "DEnKF", "LETKF", "PCE", "APCE")
CASES = (
    ("lorenz63", "Lorenz63", L63_84_SOURCE / "lorenz63_representative_seed_2026080610.npz"),
    ("lorenz84", "Lorenz84", L63_84_SOURCE / "lorenz84_representative_seed_2026080610.npz"),
    ("lorenz96", "Lorenz96", L96_SOURCE / "lorenz96_representative_seed_2026080600.npz"),
)

BASE_FONT = 7.0
TITLE_FONT = 8.2
PANEL_FONT = 8.0
ROW_FONT = 8.6


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": BASE_FONT,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def style_3d(ax: plt.Axes) -> None:
    ax.view_init(elev=24, azim=-58)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_box_aspect((1.0, 1.0, 0.82))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor("#DADADA")
        axis.line.set_color("#BDBDBD")
        axis.line.set_linewidth(0.48)


def add_header(ax: plt.Axes, panel: str, title: str) -> None:
    ax.text2D(0.0, 1.035, panel, transform=ax.transAxes, ha="left", va="baseline", fontsize=PANEL_FONT, color=PALETTE["text"])
    ax.text2D(0.14, 1.035, title, transform=ax.transAxes, ha="left", va="baseline", fontsize=TITLE_FONT, color=PALETTE["text"])


def projected(states: np.ndarray, case_key: str) -> np.ndarray:
    if case_key == "lorenz96":
        return states[:, :3]
    return states[:, :3]


def smooth_for_display(states: np.ndarray, case_key: str) -> np.ndarray:
    """Smooth only the plotted trajectory backbone; metrics still use raw simulations."""
    window = {"lorenz63": 17, "lorenz84": 15, "lorenz96": 5}[case_key]
    if window <= 1:
        return states
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(states, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.vstack(
        [np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(states.shape[1])]
    ).T
    return smoothed


def draw_panel(ax: plt.Axes, source: dict[str, np.ndarray], case_key: str, method: str | None, panel: str, title: str) -> None:
    add_header(ax, panel, title)
    style_3d(ax)
    truth = smooth_for_display(projected(source["truth_states"], case_key), case_key)[::2]
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], color=PALETTE["truth"], lw=0.86, alpha=0.88)
    if method is not None:
        estimate = smooth_for_display(projected(source[f"{method}_mean_states"], case_key), case_key)[::2]
        ax.plot(estimate[:, 0], estimate[:, 1], estimate[:, 2], color=PALETTE[method], lw=0.82, alpha=0.95)


def make_figure() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.2, 7.2))
    grid = fig.add_gridspec(3, 5, hspace=0.32, wspace=0.18)
    labels = iter("abcdefghijklmno")
    for row, (case_key, row_title, path) in enumerate(CASES):
        source = load(path)
        for col, (method, title) in enumerate(zip(METHODS, TITLES, strict=True)):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            draw_panel(ax, source, case_key, method, next(labels), title)
            if col == 0:
                ax.text2D(
                    -0.26,
                    0.50,
                    row_title,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=ROW_FONT,
                    color=PALETTE["text"],
                )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.035, top=0.965)
    base = OUTPUT / "Figure_lorenz_family_trajectory_grid"
    for suffix, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 600}, ".tiff": {"dpi": 600}}.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    svg = base.with_suffix(".svg").read_text(encoding="utf-8", errors="ignore")
    qa = {
        "layout": "3 rows x 5 columns",
        "rows": ["Lorenz63", "Lorenz84", "Lorenz96"],
        "columns": ["GT", "DEnKF", "LETKF", "PCE", "APCE"],
        "no_hilda": "HILDA" not in svg,
        "no_chinese": not any("\u4e00" <= char <= "\u9fff" for char in svg),
        "no_bold": "font-weight:bold" not in svg and "font-weight:700" not in svg,
        "editable_svg_text": svg.count("<text") > 0,
        "lorenz96_observed_states": 20,
        "trajectory_display": "moving-average smoothed for visual comparison only; metrics use raw trajectories",
    }
    (OUTPUT / "qa_manifest.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    make_figure()
