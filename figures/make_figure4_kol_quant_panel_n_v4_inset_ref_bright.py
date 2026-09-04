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
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator


HERE = Path(__file__).resolve().parent
RUN_SOURCE = (
    HERE.parent
    / "audit"
    / "figure4_kolmogorov64_velocityobs_re1500_k2_s16_t4_formal_50seeds_20260816_2gpu"
    / "source_data"
    / "kolmogorov64_velocityobs_run_source_data.csv"
)
TRACE_DIR = (
    HERE.parent
    / "audit"
    / "figure4_kolmogorov64_velocityobs_re1500_k2_t4_vorticity_seed2026081604"
    / "trace"
)
OUTPUT_BASE = HERE / "figure4_kol_quant_panel_n_v4_inset_ref_bright"

DPI = 650
FIG_W_PX = 11532
FIG_H_PX = 2112
FIG_W = FIG_W_PX / DPI
FIG_H = FIG_H_PX / DPI

# These are the detected left axes boundaries of subfigure l in the approved KSE PNG.
L_ALIGNED_LEFTS_PX = [6373, 8225, 10078]
PANEL_W = 2.20
PANEL_H = 2.00
PANEL_BOTTOM = 0.75

FONT_PANEL = 22
FONT_TITLE = 14
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
BOX_COLORS = {
    "aug_enkf": "#FF4D6D",
    "bma_static": "#7B2CBF",
    "pce": "#00A8E8",
    "apce": "#00C49A",
}


def clean_number(x: float, _pos: int | None = None) -> str:
    return f"{x:g}"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run_source() -> list[dict[str, str]]:
    with RUN_SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_values(rows: list[dict[str, str]], metric: str) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for method in METHODS:
        arr = [float(row[metric]) for row in rows if row["method"] == method and row["status"] == "completed"]
        values.append(np.asarray(arr, dtype=float))
    return values


def draw_metric_boxplot(ax: plt.Axes, rows: list[dict[str, str]], metric: str, title: str, ymax: float) -> None:
    values = grouped_values(rows, metric)
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.25},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )
    for patch, method in zip(bp["boxes"], METHODS):
        patch.set_facecolor(BOX_COLORS[method])
        patch.set_alpha(0.82)
        patch.set_edgecolor(BOX_COLORS[method])
        patch.set_linewidth(1.35)
    rng = np.random.default_rng(20260817)
    for idx, (method, arr) in enumerate(zip(METHODS, values), start=1):
        arr = arr[arr <= ymax]
        x = idx + rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(x, arr, s=20, color=BOX_COLORS[method], edgecolor="white", linewidth=0.45, alpha=0.90, zorder=3)
    ax.set_xticks(range(1, len(METHODS) + 1), [LABELS[m] for m in METHODS], rotation=38, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel(title, fontsize=FONT_AXIS)
    ax.set_title(title, fontsize=FONT_TITLE, pad=8)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FONT_TICK)


def split_velocity(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fields = np.asarray(states, dtype=np.float64).reshape(states.shape[0], 2, 64, 64)
    return fields[:, 0], fields[:, 1]


def isotropic_energy_spectrum(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ux, uy = split_velocity(states)
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


def load_trace(method: str) -> dict[str, np.ndarray]:
    path = TRACE_DIR / f"{method}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=True)
    return {key: arr[key] for key in arr.files}


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
    order_names = ["PCE", "APCE", "Ref.", "BMA", "Aug-EnKF"]
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
    inset = ax.inset_axes([0.58, 0.58, 0.36, 0.34])
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
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_xlim(14, 32)
    high_vals = np.concatenate([spec[(lam >= 14) & (lam <= 32)] for lam, spec in spectra.values()])
    high_vals = high_vals[np.isfinite(high_vals) & (high_vals > 0)]
    if high_vals.size:
        inset.set_ylim(float(np.nanmin(high_vals) * 0.65), float(np.nanmax(high_vals) * 1.55))
    inset.set_xticks([20, 30])
    inset.xaxis.set_major_formatter(FuncFormatter(clean_number))
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=FONT_TICK - 2, length=2, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#333333")
    return {method: str(TRACE_DIR / f"{method}.npz") for method in METHODS}


def add_axes_px(fig: plt.Figure, left_px: int) -> plt.Axes:
    return fig.add_axes([left_px / FIG_W_PX, PANEL_BOTTOM / FIG_H, PANEL_W / FIG_W, PANEL_H / FIG_H])


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
    rows = read_run_source()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_alpha(0.0)

    ax_nrmse = add_axes_px(fig, L_ALIGNED_LEFTS_PX[0])
    ax_crps = add_axes_px(fig, L_ALIGNED_LEFTS_PX[1])
    ax_spectrum = add_axes_px(fig, L_ALIGNED_LEFTS_PX[2])

    draw_metric_boxplot(ax_nrmse, rows, "nrmse", "nRMSE", ymax=0.08)
    draw_metric_boxplot(ax_crps, rows, "crps", "CRPS", ymax=0.06)
    trace_paths = draw_energy_spectrum(ax_spectrum)

    fig.text(
        (L_ALIGNED_LEFTS_PX[0] - round(0.14 * DPI)) / FIG_W_PX,
        0.985,
        "n",
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
        "panel": "n",
        "role": "Kolmogorov-flow quantitative reconstruction metrics and kinetic-energy spectrum",
        "figure_size_px": [FIG_W_PX, FIG_H_PX],
        "figure_size_inches": [FIG_W, FIG_H],
        "aligned_to": "subfigure l x-axis boundaries detected from approved KSE PNG",
        "l_aligned_lefts_px": L_ALIGNED_LEFTS_PX,
        "panel_size_inches": [PANEL_W, PANEL_H],
        "panel_bottom_inches": PANEL_BOTTOM,
        "run_source": str(RUN_SOURCE),
        "run_source_sha256": sha256(RUN_SOURCE),
        "trace_dir": str(TRACE_DIR),
        "trace_paths": trace_paths,
        "metrics": ["nrmse", "crps"],
        "energy_spectrum": "2D isotropic kinetic-energy spectrum from ux, uy mean-state traces; reference from the same representative seed; high-wavenumber inset added",
        "font_rules": {"panel_label": FONT_PANEL, "panel_title": FONT_TITLE, "axis_label": FONT_AXIS, "tick": FONT_TICK},
        "outputs": outputs,
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
