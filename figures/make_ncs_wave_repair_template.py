from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


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
        "font.size": 7.0,
        "axes.titlesize": 8.1,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.85,
        "legend.frameon": False,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.facecolor": "white",
    }
)

BASE_FONT = 7.0
TITLE_FONT = 7.7
PANEL_FONT = 8.0
METHOD_FONT = 8.1
LEGEND_FONT = 6.4


PALETTE = {
    "truth": "#1F1F1F",
    "denkf": "#687285",
    "letkf": "#9AA6BD",
    "pce": "#3775BA",
    "apce": "#0F4D92",
    "text": "#242424",
    "muted": "#777777",
    "light": "#E9EDF3",
    "target": "#D36B5F",
    "width": "#4B4B4B",
}

METHODS = ("denkf", "letkf", "pce", "apce")
FIELD_METHODS = ("truth", "denkf", "pce", "apce")
LABELS = {
    "truth": "Truth",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "pce": "PCE",
    "apce": "APCE",
}
FIELD_KEYS = {
    "truth": "truth_states",
    "denkf": "denkf_mean_states",
    "pce": "pce_mean_states",
    "apce": "apce_mean_states",
}

WAVE_CMAP = LinearSegmentedColormap.from_list(
    "wave_soft_balanced",
    ["#244E78", "#8DB4CF", "#F8F8F4", "#D99584", "#9F2F3B"],
    N=256,
)


def panel_header(
    ax: plt.Axes,
    label: str,
    title: str,
    x: float = 0.0,
    y: float = 1.065,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_FONT,
        fontweight="normal",
        color=PALETTE["text"],
        ha="left",
        va="baseline",
    )
    ax.text(
        x + 0.105,
        y,
        title,
        transform=ax.transAxes,
        fontsize=TITLE_FONT,
        fontweight="normal",
        color=PALETTE["text"],
        ha="left",
        va="baseline",
    )


def setup_axis(ax: plt.Axes) -> None:
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])


def set_scientific_axis(axis: plt.Axis, power_limits: tuple[int, int] = (-2, 2)) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(power_limits)
    formatter.set_useOffset(False)
    axis.set_major_formatter(formatter)
    axis.get_offset_text().set_fontsize(BASE_FONT - 0.4)
    axis.get_offset_text().set_fontweight("normal")


def format_time_label(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    exponent_value = int(exponent)
    return rf"$t={float(mantissa):.1f}\times10^{{{exponent_value}}}$"


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in raw.items():
                row[key] = value if key in {"asset", "method"} else (
                    None if value in (None, "") else float(value)
                )
            rows.append(row)
    return rows


def by_method(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] == method]


def bootstrap_ci(values: np.ndarray, seed: int = 13, n_bootstrap: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))]
    estimates = sample.mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def metric_values(rows: list[dict[str, object]], method: str, key: str, scale: float) -> np.ndarray:
    return np.asarray([float(row[key]) for row in by_method(rows, method)], dtype=float) * scale


def load_representative(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ["times", "truth_states", "denkf_mean_states", "pce_mean_states", "apce_mean_states"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Representative source data is missing keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def panel_field_plate(fig: plt.Figure, spec, source: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.055)
    times = source["times"]
    nx = source["truth_states"].shape[1] // 2
    fields = {method: source[FIELD_KEYS[method]][:, :nx] for method in FIELD_METHODS}
    vmax = float(np.nanpercentile(np.abs(np.concatenate([item.ravel() for item in fields.values()])), 99.4))
    axes = [fig.add_subplot(sub[0, i]) for i in range(4)]
    image = None
    for index, (ax, method) in enumerate(zip(axes, FIELD_METHODS, strict=True)):
        image = ax.imshow(
            fields[method],
            aspect="auto",
            origin="upper",
            extent=[0, 1, times[-1], times[0]],
            cmap=WAVE_CMAP,
            vmin=-vmax,
            vmax=vmax,
            interpolation="bicubic",
            resample=True,
        )
        for sensor in np.asarray(source["observation_indices"], dtype=int):
            ax.axvline(sensor / max(nx - 1, 1), color="white", linewidth=0.45, alpha=0.55)
        ax.set_title(LABELS[method], pad=3, fontsize=METHOD_FONT, fontweight="normal")
        ax.set_xlabel(r"$x$")
        if index == 0:
            panel_header(ax, "a", "Physical field comparison", x=-0.17, y=1.18)
            ax.set_ylabel(r"$t$")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=2.0, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(sub[0, 4])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$q(x,t)$", labelpad=4)
    cbar.ax.tick_params(labelsize=7.0, length=2.0, pad=1.5)

def panel_phase(ax: plt.Axes, source: dict[str, np.ndarray]) -> None:
    panel_header(ax, "b", "Phase-space trajectory")
    setup_axis(ax)
    nx = source["truth_states"].shape[1] // 2
    node = nx // 4
    for method in FIELD_METHODS:
        states = source[FIELD_KEYS[method]]
        q = states[:, node]
        v = states[:, nx + node]
        linewidth = 1.95 if method in {"truth", "apce"} else 1.45
        linestyle = "-" if method != "denkf" else (0, (3, 2))
        ax.plot(
            q,
            v,
            color=PALETTE[method],
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=0.93,
            label=LABELS[method],
        )
    ax.scatter(
        source["truth_states"][0, node],
        source["truth_states"][0, nx + node],
        s=28,
        color="white",
        edgecolor=PALETTE["truth"],
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        source["truth_states"][-1, node],
        source["truth_states"][-1, nx + node],
        s=30,
        color=PALETTE["target"],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\dot{q}$")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        handlelength=1.5,
        columnspacing=0.8,
        fontsize=LEGEND_FONT,
    )


def panel_profiles(fig: plt.Figure, spec, source: dict[str, np.ndarray]) -> None:
    sub = spec.subgridspec(1, 3, wspace=0.24)
    times = source["times"]
    nx = source["truth_states"].shape[1] // 2
    x = np.linspace(0.0, 1.0, nx)
    target_times = [0.25, 0.55, 1.00]
    indices = [int(np.argmin(np.abs(times - item))) for item in target_times]
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    y_stack = []
    for method in FIELD_METHODS:
        y_stack.append(source[FIELD_KEYS[method]][indices, :nx])
    y_stack_arr = np.concatenate([item.ravel() for item in y_stack])
    pad = 0.08 * (float(y_stack_arr.max()) - float(y_stack_arr.min()) + 1.0e-12)
    ylim = (float(y_stack_arr.min()) - pad, float(y_stack_arr.max()) + pad)
    for index, (ax, step) in enumerate(zip(axes, indices, strict=True)):
        if index == 0:
            panel_header(ax, "c", "Wave-profile comparison", x=-0.22, y=1.24)
            ax.set_ylabel(r"$q(x,t)$")
        else:
            ax.set_yticklabels([])
        for method in FIELD_METHODS:
            states = source[FIELD_KEYS[method]]
            linewidth = 1.9 if method in {"truth", "apce"} else 1.35
            linestyle = "-" if method != "denkf" else (0, (3, 2))
            ax.plot(
                x,
                states[step, :nx],
                color=PALETTE[method],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.94,
            )
        ax.set_xlabel(r"$x$")
        ax.set_title(format_time_label(float(times[step])), pad=4, fontweight="normal")
        ax.set_ylim(*ylim)
        setup_axis(ax)

def bar_metric(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    key: str,
    ylabel: str,
    panel: str,
    scale: float,
    title: str,
    ylim_pad: float = 0.16,
) -> None:
    panel_header(ax, panel, title)
    setup_axis(ax)
    x = np.arange(len(METHODS))
    means = np.asarray([metric_values(rows, method, key, scale).mean() for method in METHODS])
    ci = np.asarray([bootstrap_ci(metric_values(rows, method, key, scale)) for method in METHODS])
    lower = means - ci[:, 0]
    upper = ci[:, 1] - means
    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=[PALETTE[method] for method in METHODS],
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=PALETTE["text"],
        elinewidth=1.65,
        capsize=5.0,
        capthick=1.65,
        zorder=5,
    )
    rng = np.random.default_rng(17)
    for i, method in enumerate(METHODS):
        values = metric_values(rows, method, key, scale)
        jitter = rng.normal(0.0, 0.045, size=values.size)
        ax.scatter(
            np.full(values.size, i) + jitter,
            values,
            s=24,
            color=PALETTE[method],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.78,
            zorder=6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[method] for method in METHODS], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    y_max = float((means + upper).max())
    ax.set_ylim(0.0, y_max * (1.0 + ylim_pad))
    _ = bars


def panel_coverage_width(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    panel_header(ax, "f", "Calibration and width")
    setup_axis(ax)
    x = np.arange(len(METHODS))
    coverage = np.asarray([metric_values(rows, method, "coverage_90", 100.0).mean() for method in METHODS])
    coverage_ci = np.asarray([bootstrap_ci(metric_values(rows, method, "coverage_90", 100.0)) for method in METHODS])
    cov_err = np.vstack([coverage - coverage_ci[:, 0], coverage_ci[:, 1] - coverage])
    width = np.asarray([metric_values(rows, method, "interval_width_90", 100.0).mean() for method in METHODS])
    width_ci = np.asarray([bootstrap_ci(metric_values(rows, method, "interval_width_90", 100.0)) for method in METHODS])
    width_err = np.vstack([width - width_ci[:, 0], width_ci[:, 1] - width])
    ax.bar(
        x,
        coverage,
        width=0.60,
        color=[PALETTE[method] for method in METHODS],
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        x,
        coverage,
        yerr=cov_err,
        fmt="none",
        ecolor=PALETTE["text"],
        elinewidth=1.65,
        capsize=5.0,
        capthick=1.65,
        zorder=5,
    )
    ax.axhline(90.0, color=PALETTE["target"], linewidth=1.05, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[method] for method in METHODS], rotation=25, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(55.0, 101.5)
    width_axis = ax.twinx()
    width_axis.errorbar(
        x,
        width,
        yerr=width_err,
        fmt="D-",
        color=PALETTE["width"],
        ecolor=PALETTE["width"],
        elinewidth=1.15,
        capsize=3.5,
        capthick=1.15,
        markersize=4.4,
        linewidth=1.15,
        zorder=7,
    )
    width_axis.set_ylabel(r"Interval width ($10^{-2}$)", color=PALETTE["width"])
    width_axis.tick_params(axis="y", colors=PALETTE["width"], labelsize=BASE_FONT - 0.4)
    width_axis.spines["top"].set_visible(False)
    width_axis.spines["right"].set_linewidth(0.75)
    legend = [
        Line2D([0], [0], color=PALETTE["target"], lw=1.05, linestyle=(0, (3, 2)), label="90% target"),
        Line2D([0], [0], color=PALETTE["width"], marker="D", lw=1.15, markersize=4.4, label="Width"),
    ]
    ax.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.55, 1.005),
        ncol=2,
        handlelength=1.5,
        borderaxespad=0.0,
        fontsize=LEGEND_FONT,
    )


def make_figure(rows: list[dict[str, object]], source: dict[str, np.ndarray], output: Path) -> None:
    fig = plt.figure(figsize=(7.45, 6.65))
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.30, 1.05, 1.00],
        hspace=0.58,
        wspace=0.46,
    )
    panel_field_plate(fig, grid[0, :], source)
    panel_phase(fig.add_subplot(grid[1, 0:2]), source)
    panel_profiles(fig, grid[1, 2:4], source)
    bar_metric(
        fig.add_subplot(grid[2, 0]),
        rows,
        "displacement_nrmse",
        "nRMSE (%)",
        "d",
        100.0,
        "Displacement error",
    )
    bar_metric(
        fig.add_subplot(grid[2, 1]),
        rows,
        "crps",
        r"CRPS ($10^{-3}$)",
        "e",
        1000.0,
        "Probabilistic score",
    )
    panel_coverage_width(fig.add_subplot(grid[2, 2:4]), rows)

    output.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.075, top=0.93)
    base = output / "Figure_wave_repair_NCS_template"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_contract(output: Path, source_metrics: Path, representative_source: Path) -> None:
    contract = """# Figure contract

Core conclusion: APCE/PCE reduce displacement error and probabilistic forecast loss under strong cognitive model mismatch, while fixed-alpha ensemble filters retain a visible dynamical bias.

Figure archetype: asymmetric mixed-modality figure led by a physical-field comparison, followed by phase-space, profile and quantitative validation panels.

Target journal/output: Nature Computational Science; editable SVG/PDF plus 600 dpi PNG/TIFF.

Backend: Python/matplotlib only; Arial font family; no HILDA content in the figure.

Final size: 7.45 x 6.65 inches.

Panel map:
- a: Truth, DEnKF, PCE and APCE spatiotemporal displacement fields under the same frozen Wave asset.
- b: Truth, DEnKF, PCE and APCE phase-space trajectories at a fixed interior node.
- c: Truth, DEnKF, PCE and APCE displacement profiles at three time points.
- d: 5-seed displacement nRMSE with bootstrap 95% confidence intervals and paired seed dots.
- e: 5-seed CRPS with bootstrap 95% confidence intervals and paired seed dots.
- f: 90% coverage bars with interval-width markers.

Statistics: n=5 paired seeds; bars are means; error bars are paired-seed bootstrap 95% confidence intervals.

Source data: wave-repair metrics CSV and representative trajectory NPZ exported from the frozen Super-Server protocol.

Reviewer risk: this figure supports the strong-mismatch Wave gate only; Spring/Heat must pass their own gates before cross-system claims are allowed.
"""
    (output / "Figure_wave_repair_NCS_template_contract.md").write_text(contract, encoding="utf-8")
    provenance = {
        "metrics_source": source_metrics.name,
        "representative_source": representative_source.name,
        "methods_in_figure": ["Truth", "DEnKF", "LETKF", "PCE", "APCE"],
        "excluded_from_figure": ["HILDA", "HILDA-CRN-Fixed", "HILDA-KP"],
    }
    (output / "figure_source_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_metrics = root / "results_wave_repair_5seeds" / "runs.csv"
    representative_source = root / "figures" / "source_data" / "wave_repair_representative_source.npz"
    output = root / "figures" / "ncs_wave_repair_template"
    rows = [row for row in load_rows(source_metrics) if row["method"] in METHODS]
    source = load_representative(representative_source)
    make_figure(rows, source, output)
    shutil.copy2(source_metrics, output / "source_data_runs.csv")
    shutil.copy2(representative_source, output / "wave_repair_representative_source.npz")
    write_contract(output, source_metrics, representative_source)

    svg_path = output / "Figure_wave_repair_NCS_template.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    qa = {
        "rows": len(rows),
        "methods": sorted({str(row["method"]) for row in rows}),
        "figure_contains_hilda_text": "HILDA" in svg_text or "hilda" in svg_text,
        "svg_fonttype": mpl.rcParams["svg.fonttype"],
        "pdf_fonttype": mpl.rcParams["pdf.fonttype"],
        "formats": ["svg", "pdf", "png", "tiff"],
        "panel_labels": list("abcdef"),
        "svg_text_nodes": svg_text.count("<text"),
        "svg_has_chinese_text": bool(any("\u4e00" <= char <= "\u9fff" for char in svg_text)),
        "all_source_arrays_finite": bool(
            np.isfinite(
                np.concatenate(
                    [
                        source["times"].ravel(),
                        source["truth_states"].ravel(),
                        source["denkf_mean_states"].ravel(),
                        source["pce_mean_states"].ravel(),
                        source["apce_mean_states"].ravel(),
                    ]
                )
            ).all()
        ),
    }
    (output / "qa_manifest.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
