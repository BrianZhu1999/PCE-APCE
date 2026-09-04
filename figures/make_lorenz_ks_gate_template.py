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
RESULTS = ROOT / "results_lorenz_ks_gate_5seeds_l96obs20"
SOURCE_DIR = RESULTS / "representative_source"
OUTPUT = ROOT / "figures" / "lorenz_ks_gate_template"

PALETTE = {
    "truth": "#1F1F1F",
    "misspecified_forecast": "#A6A6A6",
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
    "misspecified_forecast": "Wrong forecast",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "pce": "PCE",
    "apce": "APCE",
}
FIELD_KEYS = {
    "truth": "truth_states",
    "misspecified_forecast": "misspecified_forecast_mean_states",
    "denkf": "denkf_mean_states",
    "letkf": "letkf_mean_states",
    "pce": "pce_mean_states",
    "apce": "apce_mean_states",
}

FIELD_CMAP = LinearSegmentedColormap.from_list(
    "field_soft_balanced",
    ["#244E78", "#8DB4CF", "#F8F8F4", "#D99584", "#9F2F3B"],
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
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
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


def metric_values(rows: list[dict[str, object]], case: str, method: str, key: str, scale: float) -> np.ndarray:
    return (
        np.asarray(
            [float(row[key]) for row in rows if row["case"] == case and row["method"] == method],
            dtype=float,
        )
        * scale
    )


def bootstrap_ci(values: np.ndarray, seed: int = 13, n_bootstrap: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))]
    estimates = sample.mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def load_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ["times", "coordinates", "truth_states", "denkf_mean_states", "letkf_mean_states", "pce_mean_states", "apce_mean_states"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Representative source data is missing keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def line_style(method: str) -> dict[str, object]:
    return {
        "truth": {"lw": 1.05, "ls": "-", "alpha": 0.96},
        "misspecified_forecast": {"lw": 0.86, "ls": "-", "alpha": 0.86},
        "denkf": {"lw": 0.90, "ls": (0, (3, 2)), "alpha": 0.92},
        "letkf": {"lw": 0.90, "ls": (0, (3, 2)), "alpha": 0.92},
        "pce": {"lw": 0.96, "ls": "-", "alpha": 0.94},
        "apce": {"lw": 1.00, "ls": "-", "alpha": 0.96},
    }[method]


def style_3d_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(labelsize=BASE_FONT - 0.9, width=0.5, pad=-3)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor("#DADADA")
        axis.line.set_color("#BDBDBD")
        axis.line.set_linewidth(0.5)


def plot_lorenz_3d_panel(
    ax: plt.Axes,
    truth: np.ndarray,
    method: np.ndarray | None,
    panel: str,
    title: str,
    method_color: str,
) -> None:
    ax.text2D(
        0.0,
        1.03,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=PANEL_FONT,
        fontweight="normal",
        color=PALETTE["text"],
    )
    ax.text2D(
        0.13,
        1.03,
        title,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=TITLE_FONT,
        fontweight="normal",
        color=PALETTE["text"],
    )
    style_3d_axis(ax)
    ax.view_init(elev=26, azim=-58)
    truth_line = truth[:, :3][::2]
    ax.plot(
        truth_line[:, 0],
        truth_line[:, 1],
        truth_line[:, 2],
        color=PALETTE["truth"],
        **line_style("truth"),
    )
    if method is not None:
        method_line = method[:, :3][::2]
        ax.plot(
            method_line[:, 0],
            method_line[:, 1],
            method_line[:, 2],
            color=method_color,
            lw=0.92,
            ls="-",
            alpha=0.96,
        )
    ax.scatter(
        truth_line[0, 0],
        truth_line[0, 1],
        truth_line[0, 2],
        s=10,
        color="white",
        edgecolor=PALETTE["truth"],
        linewidth=0.45,
        zorder=5,
    )
    ax.scatter(
        truth_line[-1, 0],
        truth_line[-1, 1],
        truth_line[-1, 2],
        s=14,
        color=PALETTE["truth"],
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def panel_field_plate(fig: plt.Figure, spec, source: dict[str, np.ndarray], case: str) -> None:
    nested = spec.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.08)
    times = source["times"]
    coord = source["coordinates"]
    truth = source["truth_states"]
    fields = {method: source[FIELD_KEYS[method]] for method in FIELD_METHODS}
    truth_span = float(np.nanpercentile(np.abs(truth), 99.0))
    errors = {method: np.abs(fields[method] - truth) for method in ("denkf", "pce", "apce")}
    err_max = float(np.nanpercentile(np.concatenate([value.ravel() for value in errors.values()]), 99.0))
    axes = [fig.add_subplot(nested[0, i]) for i in range(4)]
    image = None
    titles = ("Truth", "DEnKF error", "PCE error", "APCE error")
    for idx, (ax, method, title) in enumerate(zip(axes, FIELD_METHODS, titles, strict=True)):
        setup_axis(ax)
        field = truth if method == "truth" else errors[method]
        cmap = FIELD_CMAP if method == "truth" else ERROR_CMAP
        vmin = -truth_span if method == "truth" else 0.0
        vmax = truth_span if method == "truth" else err_max
        image = ax.imshow(
            field,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[coord[0], coord[-1], times[-1], times[0]],
            interpolation="nearest",
        )
        ax.set_title(title, pad=3, fontsize=METHOD_FONT, fontweight="normal")
        ax.set_xlabel(r"State index $i$" if case == "lorenz96" else r"Position $x$")
        if idx == 0:
            panel_header(
                ax,
                "a",
                "State-time field and absolute error" if case == "lorenz96" else "Physical field and absolute error",
                x=-0.22,
                y=1.13,
            )
            ax.set_ylabel(r"$t$")
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


def panel_phase(ax: plt.Axes, source: dict[str, np.ndarray], case: str) -> None:
    title = "Phase-space trajectory" if case == "lorenz96" else "Low-mode trajectory"
    panel_header(ax, "b", title)
    setup_axis(ax)
    if case == "lorenz96":
        get_xy = lambda states: (states[:, 0], states[:, 1])
        xlabel, ylabel = r"$x_0$", r"$x_1$"
    else:
        def get_xy(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            spectrum = np.fft.fft(states, axis=1) / states.shape[1]
            return spectrum[:, 1].real, spectrum[:, 2].real

        xlabel, ylabel = r"$a_1$", r"$a_2$"
    for method in FIELD_METHODS:
        x, y = get_xy(source[FIELD_KEYS[method]])
        ax.plot(x, y, color=PALETTE[method], label=LABELS[method], **line_style(method))
    x0, y0 = get_xy(source["truth_states"])
    ax.scatter(x0[0], y0[0], s=18, color="white", edgecolor=PALETTE["truth"], linewidth=0.8, zorder=4)
    ax.scatter(x0[-1], y0[-1], s=24, color=PALETTE["truth"], edgecolor="white", linewidth=0.45, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=PALETTE["light"], linewidth=0.6)
    ax.legend(loc="best", handlelength=1.6, fontsize=LEGEND_FONT)


def panel_profiles(fig: plt.Figure, spec, source: dict[str, np.ndarray], case: str) -> None:
    nested = spec.subgridspec(1, 3, wspace=0.20)
    times = source["times"]
    coord = source["coordinates"]
    indices = [int(round(frac * (len(times) - 1))) for frac in (0.30, 0.62, 1.00)]
    values = []
    for method in FIELD_METHODS:
        values.append(source[FIELD_KEYS[method]][indices, :])
    y_min, y_max = np.nanpercentile(np.concatenate([value.ravel() for value in values]), [1, 99])
    pad = 0.08 * (y_max - y_min + 1.0e-12)
    axes = [fig.add_subplot(nested[0, i]) for i in range(3)]
    for idx, (ax, step) in enumerate(zip(axes, indices, strict=True)):
        setup_axis(ax)
        for method in FIELD_METHODS:
            ax.plot(coord, source[FIELD_KEYS[method]][step], color=PALETTE[method], label=LABELS[method], **line_style(method))
        ax.set_title(format_time_label(float(times[step])), pad=3, fontsize=METHOD_FONT, fontweight="normal")
        ax.set_xlabel(r"State index $i$" if case == "lorenz96" else r"Position $x$")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(color=PALETTE["light"], linewidth=0.6)
        if idx == 0:
            panel_header(
                ax,
                "c",
                "State profiles" if case == "lorenz96" else "Field profiles",
                x=-0.22,
                y=1.13,
            )
            ax.set_ylabel(r"$x_i(t)$" if case == "lorenz96" else r"$u(x,t)$")
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
    if y_max < 1.0e-2 or y_max > 1.0e3:
        set_scientific_axis(ax.yaxis, power_limits=(-2, 2))
    _ = bars


def panel_coverage_width(ax: plt.Axes, rows: list[dict[str, object]], case: str, panel: str = "f") -> None:
    panel_header(ax, panel, "Coverage & width")
    setup_axis(ax)
    x = np.arange(len(METHODS))
    coverage = np.asarray([metric_values(rows, case, method, "coverage_90", 100.0).mean() for method in METHODS])
    coverage_ci = np.asarray([bootstrap_ci(metric_values(rows, case, method, "coverage_90", 100.0)) for method in METHODS])
    cov_err = np.vstack([coverage - coverage_ci[:, 0], coverage_ci[:, 1] - coverage])
    width = np.asarray([metric_values(rows, case, method, "interval_width_90", 1.0).mean() for method in METHODS])
    width_ci = np.asarray([bootstrap_ci(metric_values(rows, case, method, "interval_width_90", 1.0)) for method in METHODS])
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
    width_axis.set_ylabel("Interval width", color=PALETTE["width"])
    if float(width.max()) < 1.0e-2 or float(width.max()) > 1.0e3:
        set_scientific_axis(width_axis.yaxis, power_limits=(-2, 2))
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
        bbox_to_anchor=(0.52, 1.005),
        ncol=2,
        handlelength=1.5,
        borderaxespad=0.0,
        fontsize=LEGEND_FONT,
    )


def make_case_figure(rows: list[dict[str, object]], source: dict[str, np.ndarray], case: str, output: Path) -> None:
    fig = plt.figure(figsize=(7.45, 6.65))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.30, 1.05, 1.00], hspace=0.58, wspace=0.46)
    panel_field_plate(fig, grid[0, :], source, case)
    panel_phase(fig.add_subplot(grid[1, 0:2]), source, case)
    panel_profiles(fig, grid[1, 2:4], source, case)
    bar_metric(fig.add_subplot(grid[2, 0]), rows, case, "nrmse", "nRMSE (%)", "d", 100.0, "State error")
    bar_metric(fig.add_subplot(grid[2, 1]), rows, case, "crps", "CRPS", "e", 1.0, "Probabilistic score")
    panel_coverage_width(fig.add_subplot(grid[2, 2:4]), rows, case)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.075, top=0.93)
    base = output / (
        "Figure_lorenz96_gate_NCS_template" if case == "lorenz96" else "Figure_ks_gate_NCS_template"
    )
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def make_lorenz_case_figure(rows: list[dict[str, object]], source: dict[str, np.ndarray], output: Path) -> None:
    fig = plt.figure(figsize=(11.2, 6.15))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.18, 0.98], hspace=0.42, wspace=0.22)
    truth = source["truth_states"]
    panels = [
        ("a", "GT", None, PALETTE["truth"]),
        ("b", "DEnKF", source["denkf_mean_states"], PALETTE["denkf"]),
        ("c", "LETKF", source["letkf_mean_states"], PALETTE["letkf"]),
        ("d", "PCE", source["pce_mean_states"], PALETTE["pce"]),
        ("e", "APCE", source["apce_mean_states"], PALETTE["apce"]),
    ]
    for idx, (panel, title, trajectory, color) in enumerate(panels):
        ax = fig.add_subplot(grid[0, idx], projection="3d")
        plot_lorenz_3d_panel(ax, truth, trajectory, panel, title, color)
    bar_metric(fig.add_subplot(grid[1, 0]), rows, "lorenz96", "nrmse", "nRMSE (%)", "f", 100.0, "State error")
    bar_metric(fig.add_subplot(grid[1, 1]), rows, "lorenz96", "crps", "CRPS", "g", 1.0, "Probabilistic score")
    panel_coverage_width(fig.add_subplot(grid[1, 2:5]), rows, "lorenz96", "h")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.065, top=0.95)
    base = output / "Figure_lorenz96_gate_NCS_template"
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_contract(output: Path) -> None:
    text = """# Lorenz-96 / KS NCS quick-gate figure contract

Core conclusion: APCE/PCE are stress-tested on two chaotic uncertain-equation systems, with the same evidence logic used for Wave, Spring and Heat.

Evidence chain:
- a: representative state-time field and absolute-error maps show whether full fields are reconstructed under sparse observations.
- b: phase-space or low-mode trajectories test dynamical consistency rather than pointwise error only.
- c: three state/field profiles show spatial reconstruction at distinct times.
- d/e/f: five paired seeds quantify nRMSE, CRPS, 90% coverage and interval width.

Archetype: asymmetric mixed-modality figure.

Backend: Python/matplotlib only; Arial font family; all text is English and regular weight; no HILDA content.

Review risk: these are quick-gate pressure tests. They decide whether Lorenz-96 and KS should enter 20-seed confirmation; they are not final NCS claims.
"""
    (output / "Figure_lorenz_ks_gate_NCS_template_contract.md").write_text(text, encoding="utf-8")


def qa(output: Path) -> None:
    generated = [
        output / "Figure_lorenz96_gate_NCS_template.svg",
        output / "Figure_ks_gate_NCS_template.svg",
    ]
    manifest = {
        "no_hilda_in_svg": all("HILDA" not in path.read_text(encoding="utf-8", errors="ignore") for path in generated),
        "no_chinese_in_svg": all(
            not any("\u4e00" <= char <= "\u9fff" for char in path.read_text(encoding="utf-8", errors="ignore"))
            for path in generated
        ),
        "no_bold_svg": all(
            "font-weight:bold" not in path.read_text(encoding="utf-8", errors="ignore")
            and "font-weight:700" not in path.read_text(encoding="utf-8", errors="ignore")
            for path in generated
        ),
        "editable_svg_text": all(path.read_text(encoding="utf-8", errors="ignore").count("<text") > 0 for path in generated),
        "methods_in_figure": ["Truth", "DEnKF", "LETKF", "PCE", "APCE"],
        "statistical_seeds": 5,
    }
    (output / "qa_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_rows(RESULTS / "run_metrics.csv") if row["method"] in METHODS]
    lorenz_source = load_source(SOURCE_DIR / "lorenz96_representative_seed_2026080600.npz")
    ks_source = load_source(SOURCE_DIR / "ks_representative_seed_2026080600.npz")
    make_lorenz_case_figure(rows, lorenz_source, OUTPUT)
    make_case_figure(rows, ks_source, "ks", OUTPUT)
    write_contract(OUTPUT)
    qa(OUTPUT)
    print(json.dumps({"output": str(OUTPUT)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
