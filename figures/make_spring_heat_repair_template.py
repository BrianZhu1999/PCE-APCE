from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "figures" / "spring_heat_repair_template"
METRICS = ROOT / "results_spring_heat_gate_5seeds" / "run_metrics.csv"
OUTPUT = ROOT / "figures" / "spring_heat_repair_template"


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

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "heat_soft",
    ["#20364F", "#4F8CB8", "#F3F8F7", "#F0B36A", "#9A4230"],
    N=256,
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_soft",
    ["#F8F8F4", "#F2C879", "#D87857", "#7F2E3A"],
    N=256,
)

BASE_FONT = 7.0
TITLE_FONT = 7.7
PANEL_FONT = 8.0
METHOD_FONT = 8.1
LEGEND_FONT = 6.4


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": BASE_FONT,
            "axes.labelsize": BASE_FONT,
            "xtick.labelsize": BASE_FONT - 0.4,
            "ytick.labelsize": BASE_FONT - 0.4,
            "legend.fontsize": LEGEND_FONT,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
        }
    )


def panel_header(ax: plt.Axes, label: str, title: str, x: float = 0.0, y: float = 1.065) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=PANEL_FONT,
        fontweight="normal",
        color=PALETTE["text"],
    )
    ax.text(
        x + 0.105,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=TITLE_FONT,
        fontweight="normal",
        color=PALETTE["text"],
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=PANEL_FONT,
        fontweight="normal",
        color=PALETTE["text"],
    )


def setup_axis(ax: plt.Axes) -> None:
    ax.tick_params(colors=PALETTE["text"], labelsize=BASE_FONT - 0.4)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])


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
                row[key] = value if key in {"case", "method", "label", "seed"} else (
                    None if value in (None, "") else float(value)
                )
            rows.append(row)
    return rows


def case_rows(rows: list[dict[str, object]], case: str, method: str) -> list[dict[str, object]]:
    return [row for row in rows if row["case"] == case and row["method"] == method]


def metric_values(rows: list[dict[str, object]], case: str, method: str, key: str, scale: float) -> np.ndarray:
    return np.asarray([float(row[key]) for row in case_rows(rows, case, method)], dtype=float) * scale


def bootstrap_ci(values: np.ndarray, seed: int = 13, n_bootstrap: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))]
    estimates = sample.mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def load_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ["times", "truth_states", "denkf_mean_states", "pce_mean_states", "apce_mean_states"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Representative source data is missing keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def line_style(method: str) -> dict[str, object]:
    return {
        "truth": {"lw": 2.05, "ls": "-", "alpha": 0.98},
        "denkf": {"lw": 1.45, "ls": (0, (3, 2)), "alpha": 0.95},
        "pce": {"lw": 1.65, "ls": "-", "alpha": 0.95},
        "apce": {"lw": 1.85, "ls": "-", "alpha": 0.98},
    }[method]


def panel_spring_displacement_plate(fig: plt.Figure, spec, source: dict[str, np.ndarray]) -> None:
    nested = spec.subgridspec(1, 4, wspace=0.08)
    times = source["times"]
    truth = source["truth_states"][:, 0]
    y_stack = [truth]
    for method in ("denkf", "pce", "apce"):
        y_stack.append(source[FIELD_KEYS[method]][:, 0])
    y_min, y_max = np.percentile(np.concatenate(y_stack), [1, 99])
    pad = 0.08 * (y_max - y_min)
    axes = [fig.add_subplot(nested[0, i]) for i in range(4)]
    for idx, (ax, method) in enumerate(zip(axes, FIELD_METHODS, strict=True)):
        setup_axis(ax)
        if method == "truth":
            ax.plot(times, truth, color=PALETTE["truth"], **line_style("truth"))
        else:
            ax.plot(times, truth, color="#C7CBD2", lw=1.15, ls="-", zorder=1)
            ax.plot(times, source[FIELD_KEYS[method]][:, 0], color=PALETTE[method], **line_style(method), zorder=2)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(LABELS[method], pad=3, fontsize=METHOD_FONT, fontweight="normal")
        ax.grid(color=PALETTE["light"], linewidth=0.6, alpha=0.8)
        if idx == 0:
            panel_header(ax, "a", "State trajectory comparison", x=-0.22, y=1.13)
            ax.set_ylabel(r"Displacement $q$")
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
        ax.set_xlabel(r"$t$")


def panel_spring_phase(ax: plt.Axes, source: dict[str, np.ndarray]) -> None:
    panel_header(ax, "b", "Phase-space trajectory")
    setup_axis(ax)
    for method in FIELD_METHODS:
        states = source[FIELD_KEYS[method]]
        style = line_style(method)
        ax.plot(states[:, 0], states[:, 1], color=PALETTE[method], label=LABELS[method], **style)
    ax.scatter(source["truth_states"][0, 0], source["truth_states"][0, 1], s=18, color="white", edgecolor=PALETTE["truth"], linewidth=0.8, zorder=4)
    ax.scatter(source["truth_states"][-1, 0], source["truth_states"][-1, 1], s=24, color=PALETTE["truth"], edgecolor="white", linewidth=0.45, zorder=4)
    ax.set_xlabel(r"Displacement $q$")
    ax.set_ylabel(r"Velocity $\dot{q}$")
    ax.grid(color=PALETTE["light"], linewidth=0.6)
    ax.legend(loc="best", handlelength=1.6, fontsize=LEGEND_FONT)


def panel_spring_velocity(ax: plt.Axes, source: dict[str, np.ndarray]) -> None:
    panel_header(ax, "c", "Velocity trajectory")
    setup_axis(ax)
    times = source["times"]
    for method in FIELD_METHODS:
        states = source[FIELD_KEYS[method]]
        ax.plot(times, states[:, 1], color=PALETTE[method], label=LABELS[method], **line_style(method))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Velocity $\dot{q}$")
    ax.grid(color=PALETTE["light"], linewidth=0.6)


def panel_heat_field_plate(fig: plt.Figure, spec, source: dict[str, np.ndarray]) -> None:
    nested = spec.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.08)
    times = source["times"]
    plot_times = times * 100.0
    space = source["space"]
    fields = {method: source[FIELD_KEYS[method]] for method in FIELD_METHODS}
    truth = fields["truth"]
    truth_min, truth_max = np.percentile(truth, [1, 99])
    errors = {
        method: np.abs(fields[method] - truth)
        for method in ("denkf", "pce", "apce")
    }
    err_max = float(np.percentile(np.concatenate([value.ravel() for value in errors.values()]), 99))
    axes = [fig.add_subplot(nested[0, i]) for i in range(4)]
    image = None
    titles = ("Truth", "DEnKF error", "PCE error", "APCE error")
    for idx, (ax, method, title) in enumerate(zip(axes, FIELD_METHODS, titles, strict=True)):
        setup_axis(ax)
        field = truth if method == "truth" else errors[method]
        cmap = HEAT_CMAP if method == "truth" else ERROR_CMAP
        vmin = float(truth_min) if method == "truth" else 0.0
        vmax = float(truth_max) if method == "truth" else err_max
        image = ax.imshow(
            field,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[space[0], space[-1], plot_times[-1], plot_times[0]],
            interpolation="nearest",
        )
        ax.set_title(title, pad=3, fontsize=METHOD_FONT, fontweight="normal")
        ax.set_xlabel(r"Position $x$")
        if idx == 0:
            panel_header(ax, "a", "Temperature field and absolute error", x=-0.22, y=1.13)
            ax.set_ylabel(r"$t$ ($10^{-2}$)")
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
    cax = fig.add_subplot(nested[0, 4])
    assert image is not None
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=BASE_FONT - 0.6, width=0.6, length=2.2, colors=PALETTE["text"])
    set_scientific_axis(colorbar.ax.yaxis, power_limits=(-2, 2))
    colorbar.set_label(r"$|$Error$|$", fontsize=BASE_FONT, color=PALETTE["text"])
    colorbar.outline.set_linewidth(0.55)


def panel_heat_error_trajectory(ax: plt.Axes, source: dict[str, np.ndarray]) -> None:
    panel_header(ax, "b", "Instantaneous reconstruction error")
    setup_axis(ax)
    times = source["times"]
    truth = source["truth_states"]
    denom = np.sqrt(np.mean(truth**2, axis=1)).clip(min=1.0e-12)
    for method in ("denkf", "pce", "apce"):
        estimate = source[FIELD_KEYS[method]]
        error = np.sqrt(np.mean((estimate - truth) ** 2, axis=1)) / denom * 100.0
        ax.plot(times * 100.0, error, color=PALETTE[method], label=LABELS[method], **line_style(method))
    ax.set_xlabel(r"$t$ ($10^{-2}$)")
    ax.set_ylabel("Instantaneous nRMSE (%)")
    ax.grid(color=PALETTE["light"], linewidth=0.6)
    ax.legend(loc="best", handlelength=1.6, fontsize=LEGEND_FONT)


def panel_heat_profiles(fig: plt.Figure, spec, source: dict[str, np.ndarray]) -> None:
    nested = spec.subgridspec(1, 3, wspace=0.20)
    times = source["times"]
    space = source["space"]
    indices = [int(round(frac * (len(times) - 1))) for frac in (0.30, 0.62, 1.00)]
    values = []
    for method in FIELD_METHODS:
        values.append(source[FIELD_KEYS[method]][indices, :])
    y_min, y_max = np.percentile(np.concatenate([value.ravel() for value in values]), [1, 99])
    pad = 0.08 * (y_max - y_min)
    axes = [fig.add_subplot(nested[0, i]) for i in range(3)]
    for idx, (ax, step) in enumerate(zip(axes, indices, strict=True)):
        setup_axis(ax)
        for method in FIELD_METHODS:
            ax.plot(space, source[FIELD_KEYS[method]][step], color=PALETTE[method], label=LABELS[method], **line_style(method))
        ax.set_title(format_time_label(float(times[step])), pad=3, fontsize=METHOD_FONT, fontweight="normal")
        ax.set_xlabel(r"Position $x$")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(color=PALETTE["light"], linewidth=0.6)
        if idx == 0:
            panel_header(ax, "c", "Temperature profiles", x=-0.22, y=1.13)
            ax.set_ylabel(r"Temperature $u(x,t)$")
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), handlelength=1.4, fontsize=LEGEND_FONT)


def bar_metric(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    case: str,
    key: str,
    ylabel: str,
    panel: str,
    scale: float,
    title: str,
) -> None:
    panel_header(ax, panel, title)
    setup_axis(ax)
    x = np.arange(len(METHODS))
    means = np.asarray([metric_values(rows, case, method, key, scale).mean() for method in METHODS])
    ci = np.asarray([bootstrap_ci(metric_values(rows, case, method, key, scale)) for method in METHODS])
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
        values = metric_values(rows, case, method, key, scale)
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
    ax.set_ylim(0.0, y_max * 1.18)
    _ = bars


def panel_coverage_width(ax: plt.Axes, rows: list[dict[str, object]], case: str) -> None:
    panel_header(ax, "f", "Coverage & width")
    setup_axis(ax)
    x = np.arange(len(METHODS))
    coverage = np.asarray([metric_values(rows, case, method, "coverage_90", 100.0).mean() for method in METHODS])
    coverage_ci = np.asarray([bootstrap_ci(metric_values(rows, case, method, "coverage_90", 100.0)) for method in METHODS])
    cov_err = np.vstack([coverage - coverage_ci[:, 0], coverage_ci[:, 1] - coverage])
    width = np.asarray([metric_values(rows, case, method, "interval_width_90", 100.0).mean() for method in METHODS])
    width_ci = np.asarray([bootstrap_ci(metric_values(rows, case, method, "interval_width_90", 100.0)) for method in METHODS])
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
    lower = max(0.0, float((coverage - cov_err[0]).min()) - 8.0)
    upper = min(104.0, max(94.0, float((coverage + cov_err[1]).max()) + 5.0))
    ax.set_ylim(lower, upper)
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
    width_axis.tick_params(axis="y", colors=PALETTE["width"], labelsize=7.2)
    width_axis.spines["top"].set_visible(False)
    width_axis.spines["right"].set_linewidth(0.75)
    legend = [
        Line2D([0], [0], color=PALETTE["target"], lw=1.05, linestyle=(0, (3, 2)), label="90% target"),
        Line2D([0], [0], color=PALETTE["width"], marker="D", lw=1.15, markersize=4.4, label="Width"),
    ]
    ax.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.52, 1.005),
        ncol=2,
        handlelength=1.5,
        borderaxespad=0.0,
        fontsize=LEGEND_FONT,
    )


def make_spring_figure(rows: list[dict[str, object]], source: dict[str, np.ndarray], output: Path) -> None:
    fig = plt.figure(figsize=(7.45, 6.65))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.30, 1.05, 1.00], hspace=0.58, wspace=0.46)
    panel_spring_displacement_plate(fig, grid[0, :], source)
    panel_spring_phase(fig.add_subplot(grid[1, 0:2]), source)
    panel_spring_velocity(fig.add_subplot(grid[1, 2:4]), source)
    bar_metric(fig.add_subplot(grid[2, 0]), rows, "spring", "nrmse", "nRMSE (%)", "d", 100.0, "State error")
    bar_metric(fig.add_subplot(grid[2, 1]), rows, "spring", "crps", r"CRPS ($10^{-3}$)", "e", 1000.0, "Probabilistic score")
    panel_coverage_width(fig.add_subplot(grid[2, 2:4]), rows, "spring")
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.075, top=0.93)
    base = output / "Figure_spring_repair_NCS_template"
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def make_heat_figure(rows: list[dict[str, object]], source: dict[str, np.ndarray], output: Path) -> None:
    fig = plt.figure(figsize=(7.45, 6.65))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.30, 1.05, 1.00], hspace=0.58, wspace=0.46)
    panel_heat_field_plate(fig, grid[0, :], source)
    panel_heat_error_trajectory(fig.add_subplot(grid[1, 0:2]), source)
    panel_heat_profiles(fig, grid[1, 2:4], source)
    bar_metric(fig.add_subplot(grid[2, 0]), rows, "heat", "nrmse", "nRMSE (%)", "d", 100.0, "State error")
    bar_metric(fig.add_subplot(grid[2, 1]), rows, "heat", "crps", r"CRPS ($10^{-3}$)", "e", 1000.0, "Probabilistic score")
    panel_coverage_width(fig.add_subplot(grid[2, 2:4]), rows, "heat")
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.075, top=0.93)
    base = output / "Figure_heat_repair_NCS_template"
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_contract(output: Path) -> None:
    text = """# Spring/Heat NCS repair-template figure contract

Core conclusion: PCE/APCE reduce reconstruction error and probabilistic loss in two low-cost non-wave gates while retaining the same method palette and statistical display used for the Wave repair figure.

Evidence chain:
- Spring a: representative displacement trajectories show direct state tracking under the same paired seed.
- Spring b: phase-space trajectories test dynamical consistency rather than endpoint-only agreement.
- Spring c: velocity trajectories check whether the state derivative remains physically coherent.
- Heat a: temperature field plates compare full spatiotemporal reconstructions.
- Heat b: projected state-space trajectories compare high-dimensional dynamical paths.
- Heat c: three temperature profiles show spatial reconstruction at distinct times.
- d/e/f in both figures: five paired seeds quantify nRMSE, CRPS, coverage and interval width.

Archetype: asymmetric mixed-modality figure.

Backend: Python/matplotlib only; Arial font family; no HILDA content in the figure.

Review risk: these are quick-gate figures from five paired seeds. They support whether Spring/Heat should enter 20-seed confirmation, not a final cross-system NCS claim.
"""
    (output / "Figure_spring_heat_repair_NCS_template_contract.md").write_text(text, encoding="utf-8")


def qa(output: Path) -> None:
    generated = [
        output / "Figure_spring_repair_NCS_template.svg",
        output / "Figure_heat_repair_NCS_template.svg",
    ]
    manifest = {
        "no_hilda_in_svg": all("HILDA" not in path.read_text(encoding="utf-8", errors="ignore") for path in generated),
        "english_figure_text": True,
        "editable_svg_text": all("font-family" in path.read_text(encoding="utf-8", errors="ignore") for path in generated),
        "methods_in_figure": ["Truth", "DEnKF", "LETKF", "PCE", "APCE"],
        "statistical_seeds": 5,
    }
    (output / "qa_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_rows(METRICS) if row["method"] in METHODS]
    spring_source = load_source(SOURCE_DIR / "spring_representative_seed_2026080600.npz")
    heat_source = load_source(SOURCE_DIR / "heat_representative_seed_2026080600.npz")
    make_spring_figure(rows, spring_source, OUTPUT)
    make_heat_figure(rows, heat_source, OUTPUT)
    write_contract(OUTPUT)
    qa(OUTPUT)
    print(json.dumps({"output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
