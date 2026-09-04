from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_lorenz63_84_gate_5seeds"
SOURCE = RESULTS / "representative_source"
OUTPUT = ROOT / "figures" / "lorenz63_84_gate_template"

PALETTE = {
    "truth": "#E8898F",
    "denkf": "#7D8CB4",
    "letkf": "#A9B4CC",
    "pce": "#4F7FC1",
    "apce": "#164F91",
    "text": "#242424",
    "light": "#E9EDF3",
    "target": "#D36B5F",
    "width": "#4B4B4B",
}
METHODS = ("denkf", "letkf", "pce", "apce")
LABELS = {"truth": "GT", "denkf": "DEnKF", "letkf": "LETKF", "pce": "PCE", "apce": "APCE"}

BASE_FONT = 7.0
TITLE_FONT = 7.7
PANEL_FONT = 8.0
LEGEND_FONT = 6.2


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
            "axes.labelsize": BASE_FONT,
            "xtick.labelsize": BASE_FONT - 0.5,
            "ytick.labelsize": BASE_FONT - 0.5,
            "legend.fontsize": LEGEND_FONT,
            "axes.linewidth": 0.72,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_rows() -> list[dict[str, object]]:
    rows = []
    with (RESULTS / "run_metrics.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                row[key] = value if key in {"case", "method", "label", "seed"} else float(value)
            rows.append(row)
    return rows


def load_source(case: str) -> dict[str, np.ndarray]:
    with np.load(SOURCE / f"{case}_representative_seed_2026080610.npz") as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metric_values(rows: list[dict[str, object]], case: str, method: str, key: str, scale: float = 1.0) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows if row["case"] == case and row["method"] == method]) * scale


def bootstrap_ci(values: np.ndarray, seed: int = 23, n_boot: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def panel_label_3d(ax: plt.Axes, label: str, title: str) -> None:
    ax.text2D(0.0, 1.035, label, transform=ax.transAxes, ha="left", va="baseline", fontsize=PANEL_FONT, color=PALETTE["text"])
    ax.text2D(0.14, 1.035, title, transform=ax.transAxes, ha="left", va="baseline", fontsize=TITLE_FONT, color=PALETTE["text"])


def style_3d(ax: plt.Axes) -> None:
    ax.view_init(elev=24, azim=-58)
    ax.grid(False)
    ax.tick_params(labelsize=BASE_FONT - 1.0, width=0.5, pad=-3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1.0, 1.0, 0.82))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor("#DADADA")
        axis.line.set_color("#BDBDBD")
        axis.line.set_linewidth(0.5)


def plot_trajectory(ax: plt.Axes, source: dict[str, np.ndarray], method: str | None, label: str, title: str) -> None:
    panel_label_3d(ax, label, title)
    style_3d(ax)
    truth = source["truth_states"][::2]
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], color=PALETTE["truth"], lw=0.92, alpha=0.92)
    if method is not None:
        states = source[f"{method}_mean_states"][::2]
        ax.plot(states[:, 0], states[:, 1], states[:, 2], color=PALETTE[method], lw=0.86, alpha=0.95)


def panel_header(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.07, label, transform=ax.transAxes, ha="left", va="baseline", fontsize=PANEL_FONT, color=PALETTE["text"])
    ax.text(0.07, 1.07, title, transform=ax.transAxes, ha="left", va="baseline", fontsize=TITLE_FONT, color=PALETTE["text"])


def metric_panel(ax: plt.Axes, rows: list[dict[str, object]], case: str, key: str, ylabel: str, label: str, title: str, scale: float = 1.0) -> None:
    panel_header(ax, label, title)
    x = np.arange(len(METHODS))
    means = np.asarray([metric_values(rows, case, method, key, scale).mean() for method in METHODS])
    ci = np.asarray([bootstrap_ci(metric_values(rows, case, method, key, scale), seed=41 + i) for i, method in enumerate(METHODS)])
    lower = means - ci[:, 0]
    upper = ci[:, 1] - means
    ax.bar(x, means, width=0.58, color=[PALETTE[m] for m in METHODS], edgecolor="white", linewidth=0.65, zorder=2)
    ax.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor=PALETTE["text"], elinewidth=1.1, capsize=3.8, capthick=1.1, zorder=5)
    rng = np.random.default_rng(11)
    for i, method in enumerate(METHODS):
        values = metric_values(rows, case, method, key, scale)
        ax.scatter(np.full(values.size, i) + rng.normal(0, 0.04, values.size), values, s=18, color=PALETTE[method], edgecolor="white", linewidth=0.35, alpha=0.75, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=PALETTE["light"], linewidth=0.55, zorder=0)
    ax.set_ylim(0.0, float((means + upper).max()) * 1.18)


def make_figure() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    sources = {"lorenz63": load_source("lorenz63"), "lorenz84": load_source("lorenz84")}
    fig = plt.figure(figsize=(12.4, 7.2))
    grid = fig.add_gridspec(3, 5, height_ratios=[1.08, 1.08, 0.92], hspace=0.44, wspace=0.26)
    labels = iter("abcdefghijklmnop")
    for row_idx, case in enumerate(("lorenz63", "lorenz84")):
        source = sources[case]
        titles = ["GT", "DEnKF", "LETKF", "PCE", "APCE"]
        methods: list[str | None] = [None, "denkf", "letkf", "pce", "apce"]
        for col_idx, (title, method) in enumerate(zip(titles, methods, strict=True)):
            ax = fig.add_subplot(grid[row_idx, col_idx], projection="3d")
            plot_trajectory(ax, source, method, next(labels), title if col_idx > 0 else f"{case}: {title}")
    metric_panel(fig.add_subplot(grid[2, 0]), rows, "lorenz63", "nrmse", "nRMSE (%)", "k", "Lorenz63 error", scale=100.0)
    metric_panel(fig.add_subplot(grid[2, 1]), rows, "lorenz63", "crps", "CRPS", "l", "Lorenz63 CRPS")
    metric_panel(fig.add_subplot(grid[2, 2]), rows, "lorenz84", "nrmse", "nRMSE (%)", "m", "Lorenz84 error", scale=100.0)
    metric_panel(fig.add_subplot(grid[2, 3]), rows, "lorenz84", "crps", "CRPS", "n", "Lorenz84 CRPS")
    legend_ax = fig.add_subplot(grid[2, 4])
    legend_ax.axis("off")
    handles = [Line2D([0], [0], color=PALETTE["truth"], lw=1.0, label="GT")] + [
        Line2D([0], [0], color=PALETTE[m], lw=1.0, label=LABELS[m]) for m in METHODS
    ]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, handlelength=2.0)
    legend_ax.text(0.0, 0.98, "Two observed variables per system", ha="left", va="top", fontsize=TITLE_FONT, color=PALETTE["text"], transform=legend_ax.transAxes)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.065, top=0.955)
    base = OUTPUT / "Figure_lorenz63_84_gate_NCS_template"
    for suffix, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 600}, ".tiff": {"dpi": 600}}.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)
    qa = {
        "no_hilda": "HILDA" not in base.with_suffix(".svg").read_text(encoding="utf-8", errors="ignore"),
        "no_chinese": not any("\u4e00" <= ch <= "\u9fff" for ch in base.with_suffix(".svg").read_text(encoding="utf-8", errors="ignore")),
        "editable_svg_text": base.with_suffix(".svg").read_text(encoding="utf-8", errors="ignore").count("<text") > 0,
        "backend": "python/matplotlib",
    }
    (OUTPUT / "qa_manifest.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    make_figure()
