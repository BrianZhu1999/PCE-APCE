from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator


HERE = Path(__file__).resolve().parent
RESULT_ROOT = (
    HERE.parent
    / "audit"
    / "figure4_kolmogorov64_velocityobs_re1500_k2_s16_t4_blackout40_formal_50seeds_20260816_2gpu"
)
SOURCE_DIR = RESULT_ROOT / "source_data"
TRACE_DIR = RESULT_ROOT / "traces"
RUN_SOURCE = SOURCE_DIR / "kolmogorov64_blackout_run_source_data.csv"
LEAD_SOURCE = SOURCE_DIR / "kolmogorov64_blackout_lead_time_source_data.csv"
OUTPUT_BASE = HERE / "figure4_kol_forecast_panels_op_v5_bottomcbar_midgap07_metriccolors_cbarh15_labeltop"

DPI = 650
FIG_W_PX = 11532
FIG_H_PX = 2112
FIG_W = FIG_W_PX / DPI
FIG_H = FIG_H_PX / DPI

FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11

METHODS = ["aug_enkf", "bma_static", "pce", "apce"]
LABELS = {"aug_enkf": "Aug-EnKF", "bma_static": "BMA", "pce": "PCE", "apce": "APCE"}
LINE_COLORS = {
    "aug_enkf": "#A8B0B7",
    "bma_static": "#D7A64A",
    "pce": "#4E79A7",
    "apce": "#59A14F",
    "truth": "#111111",
}
METRIC_COLORS = {
    "aug_enkf": "#BFC3C7",
    "bma_static": "#B8CCE0",
    "pce": "#D94A5A",
    "apce": "#00A887",
}

REP_SEED = 2026081637
SNAPSHOT_STEPS = (45, 55)
BLACKOUT_START_STEP = 40
VORTICITY_LIMIT = 8.0

# Match m-panel geometry exactly, but place it on the full-width transparent row.
IMAGE_W = 1.80
IMAGE_H = 1.75
IMAGE_Y = 0.78
X0 = (16.428 - 0.264) / 72.0
X_GAP = 0.24
MIDDLE_GAP = 0.70
IMAGE_XS = [
    X0,
    X0 + IMAGE_W + X_GAP,
    X0 + 2 * IMAGE_W + X_GAP + MIDDLE_GAP,
    X0 + 3 * IMAGE_W + 2 * X_GAP + MIDDLE_GAP,
]

# Detected left boundaries of the approved KSE l/n quantitative columns.
P_LEFTS_PX = [6373, 8225, 10078]
L_PANEL_LABEL_X_PX = 5965
PANEL_W = 2.20
PANEL_H = 2.00
PANEL_BOTTOM = 0.75


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
        }
    )


def clean_number(x: float, _pos: int | None = None) -> str:
    return f"{x:g}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def trace_path(method: str, seed: int = REP_SEED) -> Path:
    return TRACE_DIR / f"kol64_re1500_k2_s16_t4_blackout40_{method}_seed{seed}.npz"


def load_trace(method: str) -> dict[str, np.ndarray]:
    path = trace_path(method)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def split_velocity(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fields = np.asarray(states, dtype=np.float64).reshape(states.shape[0], 2, 64, 64)
    return fields[:, 0], fields[:, 1]


def vorticity(states: np.ndarray) -> np.ndarray:
    ux, uy = split_velocity(states)
    wave = 2.0 * np.pi * np.fft.fftfreq(64, d=2.0 * np.pi / 64.0)
    ux_hat = np.fft.fft2(ux, axes=(-2, -1))
    uy_hat = np.fft.fft2(uy, axes=(-2, -1))
    dv_dx = np.fft.ifft2(1j * wave[None, :, None] * uy_hat, axes=(-2, -1)).real
    du_dy = np.fft.ifft2(1j * wave[None, None, :] * ux_hat, axes=(-2, -1)).real
    return dv_dx - du_dy


def field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("kol_vorticity", ["#214c78", "#f4f2ee", "#9d3c35"], N=256)


def add_axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def add_axes_px(fig: plt.Figure, left_px: int) -> plt.Axes:
    return fig.add_axes([left_px / FIG_W_PX, PANEL_BOTTOM / FIG_H, PANEL_W / FIG_W, PANEL_H / FIG_H])


def draw_vorticity_forecast(fig: plt.Figure) -> dict[str, str | float]:
    trace = load_trace("apce")
    truth = vorticity(np.asarray(trace["truth"], dtype=np.float64))
    apce = vorticity(np.asarray(trace["mean_states"], dtype=np.float64))
    times = np.asarray(trace["times"], dtype=np.float64)
    cmap = field_cmap().reversed()
    norm = Normalize(vmin=-VORTICITY_LIMIT, vmax=VORTICITY_LIMIT)
    columns = (
        (SNAPSHOT_STEPS[0], r"Ref.    $t=45$", truth),
        (SNAPSHOT_STEPS[0], r"APCE    $t=45$", apce),
        (SNAPSHOT_STEPS[1], r"Ref.    $t=55$", truth),
        (SNAPSHOT_STEPS[1], r"APCE    $t=55$", apce),
    )
    image = None
    for col, (step, _label, field) in enumerate(columns):
        ax = add_axes_inches(fig, IMAGE_XS[col], IMAGE_Y, IMAGE_W, IMAGE_H)
        image = ax.imshow(
            field[step].T,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            aspect="auto",
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for col, (_step, label, _field) in enumerate(columns):
        fig.text(
            (IMAGE_XS[col] + IMAGE_W / 2.0) / FIG_W,
            2.57 / FIG_H,
            label,
            ha="center",
            va="bottom",
            fontsize=FONT_TITLE,
            color="#111111",
        )
    fig.text(0.012, 0.985, "o", ha="left", va="top", fontsize=FONT_PANEL, fontweight="bold", color="#111111")
    fig.text(
        (IMAGE_XS[0] + (IMAGE_XS[-1] + IMAGE_W - IMAGE_XS[0]) / 2.0) / FIG_W,
        0.945,
        r"Vorticity forecast    $k=2,\ \mathrm{Re}=1500$",
        ha="center",
        va="top",
        fontsize=FONT_TITLE,
        color="#111111",
    )
    if image is None:
        raise RuntimeError("No forecast images were drawn")
    cbar_x = IMAGE_XS[0]
    cbar_w = IMAGE_XS[-1] + IMAGE_W - IMAGE_XS[0]
    cbar_y = 0.43
    cbar_h = 0.15
    cax = add_axes_inches(fig, cbar_x, cbar_y, cbar_w, cbar_h)
    colorbar = fig.colorbar(image, cax=cax, orientation="horizontal", extend="both", extendfrac=0.030)
    colorbar.set_ticks([-VORTICITY_LIMIT, VORTICITY_LIMIT])
    colorbar.set_ticklabels(["-8", "8"])
    colorbar.ax.tick_params(labelsize=FONT_TICK, length=0, pad=2)
    colorbar.outline.set_visible(False)
    label_gap_inches = 0.02
    fig.text(
        (cbar_x + 0.5 * cbar_w) / FIG_W,
        (cbar_y - label_gap_inches) / FIG_H,
        "Vorticity",
        ha="center",
        va="top",
        fontsize=FONT_AXIS,
        color="#111111",
    )
    displayed_vorticity_nrmse = float(
        np.sqrt(np.sum((apce[list(SNAPSHOT_STEPS)] - truth[list(SNAPSHOT_STEPS)]) ** 2) / np.sum(truth[list(SNAPSHOT_STEPS)] ** 2))
    )
    return {
        "apce_trace": str(trace_path("apce")),
        "displayed_vorticity_nrmse": displayed_vorticity_nrmse,
        "time_t45": float(times[45]),
        "time_t55": float(times[55]),
        "colorbar_position_inches": [cbar_x, cbar_y, cbar_w, cbar_h],
    }


def grouped_lead_rows() -> dict[str, list[dict[str, str]]]:
    rows = read_csv(LEAD_SOURCE)
    groups: dict[str, list[dict[str, str]]] = {method: [] for method in METHODS}
    for row in rows:
        method = row["method"]
        if method in groups:
            groups[method].append(row)
    for method in groups:
        groups[method].sort(key=lambda r: int(r["lead_index"]))
    return groups


def draw_metric_curve(ax: plt.Axes, grouped: dict[str, list[dict[str, str]]], metric: str, title: str, ylabel: str) -> None:
    styles = {
        "aug_enkf": (METRIC_COLORS["aug_enkf"], 1.90, 0.98, 2),
        "bma_static": (METRIC_COLORS["bma_static"], 1.95, 0.98, 3),
        "pce": (METRIC_COLORS["pce"], 2.40, 1.00, 4),
        "apce": (METRIC_COLORS["apce"], 2.60, 1.00, 5),
    }
    for method in METHODS:
        rows = grouped[method]
        x = np.asarray([BLACKOUT_START_STEP + int(row["lead_index"]) for row in rows], dtype=float)
        y = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        color, lw, alpha, zorder = styles[method]
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=LABELS[method], zorder=zorder)
    ax.set_xlim(40, 58)
    ax.set_xticks([40, 50, 58])
    ax.set_xlabel("Time step", fontsize=FONT_AXIS, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS, labelpad=3)
    ax.set_title(title, fontsize=FONT_TITLE, pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(
        loc="upper left",
        fontsize=FONT_TICK,
        ncol=1,
        handlelength=1.15,
        labelspacing=0.20,
        borderaxespad=0.25,
    )


def draw_pce_apce_metric_inset(fig: plt.Figure, main_left_px: int, grouped: dict[str, list[dict[str, str]]], metric: str) -> None:
    left_in = main_left_px / DPI + PANEL_W + 0.03
    bottom_in = PANEL_BOTTOM + 1.15
    inset = add_axes_inches(fig, left_in, bottom_in, 0.36, 0.50)
    for method in ["pce", "apce"]:
        rows = grouped[method]
        x = np.asarray([BLACKOUT_START_STEP + int(row["lead_index"]) for row in rows], dtype=float)
        y = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        inset.plot(x, y, color=METRIC_COLORS[method], linewidth=1.55, alpha=1.0)
    inset.set_xlim(48, 58)
    inset.set_xticks([])
    vals = []
    for method in ["pce", "apce"]:
        rows = grouped[method]
        vals.extend(float(row[f"{metric}_mean"]) for row in rows if 48 <= BLACKOUT_START_STEP + int(row["lead_index"]) <= 58)
    arr = np.asarray(vals, dtype=float)
    if arr.size:
        pad = max((float(np.nanmax(arr)) - float(np.nanmin(arr))) * 0.22, 0.002)
        inset.set_ylim(float(np.nanmin(arr) - pad), float(np.nanmax(arr) + pad))
    inset.xaxis.set_major_formatter(FuncFormatter(clean_number))
    inset.yaxis.set_major_formatter(FuncFormatter(clean_number))
    inset.set_yticks([])
    inset.tick_params(axis="both", length=0, labelbottom=False, labelleft=False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#333333")
    inset.set_facecolor("white")


def isotropic_energy_spectrum(states: np.ndarray, start_step: int = BLACKOUT_START_STEP + 1) -> tuple[np.ndarray, np.ndarray]:
    ux, uy = split_velocity(np.asarray(states, dtype=np.float64)[start_step:])
    n = ux.shape[-1]
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)
    shell = np.rint(kr).astype(int)
    max_shell = n // 2
    ux_hat = np.fft.fft2(ux, axes=(-2, -1))
    uy_hat = np.fft.fft2(uy, axes=(-2, -1))
    energy_density = 0.5 * (np.abs(ux_hat) ** 2 + np.abs(uy_hat) ** 2)
    spectrum = np.zeros(max_shell + 1, dtype=float)
    counts = np.zeros(max_shell + 1, dtype=float)
    for shell_id in range(max_shell + 1):
        mask = shell == shell_id
        if np.any(mask):
            spectrum[shell_id] = float(energy_density[:, mask].mean())
            counts[shell_id] = float(mask.sum())
    lam = np.arange(max_shell + 1, dtype=float)
    valid = (lam >= 1) & (counts > 0)
    return lam[valid], spectrum[valid]


def draw_energy_spectrum(ax: plt.Axes) -> dict[str, str]:
    traces = {method: load_trace(method) for method in METHODS}
    truth = np.asarray(traces["apce"]["truth"], dtype=float)
    lam_ref, spec_ref = isotropic_energy_spectrum(truth)
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {"Ref.": (lam_ref, spec_ref)}
    ax.plot(lam_ref, spec_ref, color=LINE_COLORS["truth"], linewidth=2.4, linestyle="--", label="Ref.", zorder=10)
    for method in METHODS:
        lam, spec = isotropic_energy_spectrum(np.asarray(traces[method]["mean_states"], dtype=float))
        spectra[LABELS[method]] = (lam, spec)
        lw = 2.35 if method in {"pce", "apce"} else 1.65
        alpha = 1.0 if method in {"pce", "apce"} else 0.82
        ax.plot(lam, spec, color=LINE_COLORS[method], linewidth=lw, alpha=alpha, label=LABELS[method])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 32)
    ax.set_ylim(1, 3.0e6)
    ax.set_xticks([10])
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.set_yticks([1, 1.0e2, 1.0e4, 1.0e6])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xlabel(r"Wavenumber $\lambda$", fontsize=FONT_AXIS)
    ax.set_ylabel(r"$E(\lambda)$", fontsize=FONT_AXIS)
    ax.set_title("Energy spectrum", fontsize=FONT_TITLE, pad=8)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order_names = ["Ref.", "PCE", "APCE", "BMA", "Aug-EnKF"]
    order = [labels.index(name) for name in order_names if name in labels]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower left",
        fontsize=FONT_TICK,
        ncol=1,
        handlelength=1.15,
        borderaxespad=0.25,
        labelspacing=0.18,
    )
    inset = ax.inset_axes([0.64, 0.53, 0.32, 0.43])
    style_map = {
        "Ref.": (LINE_COLORS["truth"], "--", 2.0, 10, 1.0),
        "Aug-EnKF": (LINE_COLORS["aug_enkf"], "-", 1.25, 2, 0.78),
        "BMA": (LINE_COLORS["bma_static"], "-", 1.35, 3, 0.82),
        "PCE": (LINE_COLORS["pce"], "-", 1.75, 5, 1.0),
        "APCE": (LINE_COLORS["apce"], "-", 1.85, 6, 1.0),
    }
    for label in ["Aug-EnKF", "BMA", "PCE", "APCE", "Ref."]:
        lam_i, spec_i = spectra[label]
        color, ls, lw, z, alpha = style_map[label]
        inset.plot(lam_i, spec_i, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=z)
    inset.set_yscale("log")
    inset.set_xlim(20, 25)
    high_vals = np.concatenate([spec[(lam >= 20) & (lam <= 25)] for lam, spec in spectra.values()])
    high_vals = high_vals[np.isfinite(high_vals) & (high_vals > 0)]
    if high_vals.size:
        inset.set_ylim(float(np.nanmin(high_vals) * 0.65), float(np.nanmax(high_vals) * 1.55))
    inset.set_xticks([20, 25])
    inset.xaxis.set_major_formatter(FuncFormatter(clean_number))
    inset.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    inset.set_yticks([])
    inset.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    inset.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    inset.yaxis.set_major_locator(mpl.ticker.NullLocator())
    inset.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    inset.tick_params(axis="x", labelsize=FONT_TICK - 2, length=2, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#333333")
    return {method: str(trace_path(method)) for method in METHODS}


def save_all(fig: plt.Figure) -> dict[str, str]:
    outputs = {
        "png": OUTPUT_BASE.with_suffix(".png"),
        "pdf": OUTPUT_BASE.with_suffix(".pdf"),
        "svg": OUTPUT_BASE.with_suffix(".svg"),
        "tiff": OUTPUT_BASE.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=DPI, transparent=True)
    fig.savefig(outputs["pdf"], transparent=True)
    fig.savefig(outputs["svg"], transparent=True)
    fig.savefig(outputs["tiff"], dpi=DPI, transparent=True)
    return {name: str(path) for name, path in outputs.items()}


def main() -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_alpha(0.0)

    vorticity_info = draw_vorticity_forecast(fig)
    label_gap_inches = float(vorticity_info.get("colorbar_label_gap_inches", 0.012))
    grouped = grouped_lead_rows()
    ax_nrmse = add_axes_px(fig, P_LEFTS_PX[0])
    ax_crps = add_axes_px(fig, P_LEFTS_PX[1])
    ax_spectrum = add_axes_px(fig, P_LEFTS_PX[2])
    draw_metric_curve(ax_nrmse, grouped, "nrmse", "nRMSE", "nRMSE")
    draw_metric_curve(ax_crps, grouped, "crps", "CRPS", "CRPS")
    trace_paths = draw_energy_spectrum(ax_spectrum)
    fig.text(
        L_PANEL_LABEL_X_PX / FIG_W_PX,
        0.985,
        "p",
        ha="left",
        va="top",
        fontsize=FONT_PANEL,
        fontweight="bold",
        color="#111111",
    )

    outputs = save_all(fig)
    plt.close(fig)
    qa = {
        "backend": "Python/matplotlib",
        "panels": ["o", "p"],
        "role": "Kolmogorov-flow blackout forecast vorticity snapshots, forecast errors and forecast-window kinetic-energy spectrum",
        "figure_size_px": [FIG_W_PX, FIG_H_PX],
        "figure_size_inches": [FIG_W, FIG_H],
        "representative_seed": REP_SEED,
        "representative_seed_selection": "minimum APCE forecast nRMSE among 50 paired blackout-forecast seeds; also minimum APCE forecast vorticity nRMSE",
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "blackout_start_step": BLACKOUT_START_STEP,
        "small_image_size_inches": [IMAGE_W, IMAGE_H],
        "small_image_size_px": [round(IMAGE_W * DPI), round(IMAGE_H * DPI)],
        "metric_panel_lefts_px": P_LEFTS_PX,
        "l_aligned_panel_label_x_px": L_PANEL_LABEL_X_PX,
        "metric_panel_size_inches": [PANEL_W, PANEL_H],
        "run_source": str(RUN_SOURCE),
        "run_source_sha256": sha256(RUN_SOURCE),
        "lead_source": str(LEAD_SOURCE),
        "lead_source_sha256": sha256(LEAD_SOURCE),
        "trace_paths": trace_paths,
        "vorticity_info": vorticity_info,
        "colorbar_label_gap_inches": label_gap_inches,
        "forecast_energy_spectrum": "2D isotropic kinetic-energy spectrum computed only from steps 41-58 after blackout",
        "font_rules": {"panel_label": FONT_PANEL, "panel_title": FONT_TITLE, "legend": FONT_LEGEND, "axis_label": FONT_AXIS, "tick": FONT_TICK},
        "outputs": outputs,
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
