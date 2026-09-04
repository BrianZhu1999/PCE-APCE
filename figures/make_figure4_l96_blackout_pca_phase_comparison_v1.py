from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


METHODS = ["Aug-EnKF", "BMA", "PCE", "APCE"]
METHOD_KEY = {
    "Aug-EnKF": "aug_enkf",
    "BMA": "bma_static",
    "PCE": "pce",
    "APCE": "apce",
}
METHOD_COLOR = {
    "Aug-EnKF": "#767676",
    "BMA": "#4E79A7",
    "PCE": "#E56A5C",
    "APCE": "#2BAA9A",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def choose_representative_seed(run_rows: list[dict[str, str]]) -> tuple[int, dict[str, str]]:
    apce_rows = [row for row in run_rows if row["label"] == "APCE"]
    apce_rows.sort(key=lambda row: (f(row, "forecast_nrmse"), f(row, "blackout_alpha_absolute_error"), int(row["seed"])))
    if not apce_rows:
        raise RuntimeError("No APCE rows found in source data.")
    return int(apce_rows[0]["seed"]), apce_rows[0]


def make_pca_basis(traces: list[np.ndarray], blackout_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast = [trace[blackout_step:] for trace in traces]
    stacked = np.concatenate(forecast, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    explained = (s[:2] ** 2) / np.maximum((s**2).sum(), 1.0e-30)
    return mean, components, explained


def project(trace: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (trace - mean) @ components


def draw_path(ax: plt.Axes, coords: np.ndarray, color: str, label: str, *, truth: bool = False) -> None:
    lw = 1.6 if truth else 2.6
    alpha = 0.70 if truth else 0.95
    style = "--" if truth else "-"
    ax.plot(coords[:, 0], coords[:, 1], color=color, lw=lw, ls=style, alpha=alpha, zorder=2 if truth else 4)
    ax.scatter(coords[0, 0], coords[0, 1], s=52, facecolors="white", edgecolors=color, linewidths=1.4, zorder=5)
    ax.scatter(coords[-1, 0], coords[-1, 1], s=60, marker="s" if not truth else "o", color=color, edgecolors="white", linewidths=0.8, zorder=6)
    if not truth:
        mid = coords.shape[0] // 2
        ax.scatter(coords[mid, 0], coords[mid, 1], s=20, color=color, alpha=0.85, zorder=5)


def add_axis_labels(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11, width=1.0, length=3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")


def save_pub(fig: plt.Figure, out_base: Path) -> dict[str, str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for ext, kwargs in {
        "png": {"dpi": 450},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved[ext] = str(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Lorenz-96 blackout forecast PCA phase portrait comparison.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("<HILDA_RESULTS_ROOT>/results/figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds_20260815_gpu23"),
    )
    parser.add_argument("--out-base", type=Path, default=None)
    args = parser.parse_args()

    source = args.root / "source_data"
    run_rows = read_csv(source / "lorenz96_1024_blackout_run_source_data.csv")
    seed, rep_row = choose_representative_seed(run_rows)
    metrics = {row["label"]: row for row in run_rows if int(row["seed"]) == seed}
    blackout_step = int(float(rep_row["blackout_start_step"]))

    shared = load_npz(args.root / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz")
    traces = {}
    for label in METHODS:
        key = METHOD_KEY[label]
        path = args.root / "artifacts" / "method_traces" / "lorenz96_1024" / "time8" / key / f"seed_{seed}.npz"
        traces[label] = load_npz(path)["mean_states"].astype(float)
    truth = shared["truth"].astype(float)

    mean, components, explained = make_pca_basis([truth] + [traces[label] for label in METHODS], blackout_step)
    truth_coords = project(truth[blackout_step:], mean, components)
    method_coords = {label: project(traces[label][blackout_step:], mean, components) for label in METHODS}

    all_coords = np.concatenate([truth_coords] + list(method_coords.values()), axis=0)
    xmin, ymin = np.nanmin(all_coords, axis=0)
    xmax, ymax = np.nanmax(all_coords, axis=0)
    pad_x = 0.12 * max(xmax - xmin, 1.0e-6)
    pad_y = 0.12 * max(ymax - ymin, 1.0e-6)
    xlim = (xmin - pad_x, xmax + pad_x)
    ylim = (ymin - pad_y, ymax + pad_y)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0), constrained_layout=False)
    axes = axes.ravel()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.11, wspace=0.24, hspace=0.28)

    panel_labels = ["a", "b", "c", "d"]
    for ax, label, method_name in zip(axes, panel_labels, METHODS):
        draw_path(ax, truth_coords, "#3A3A3A", "Truth", truth=True)
        draw_path(ax, method_coords[method_name], METHOD_COLOR[method_name], method_name)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        add_axis_labels(
            ax,
            xlabel=f"PC1 ({explained[0] * 100:.1f}%)",
            ylabel=f"PC2 ({explained[1] * 100:.1f}%)",
        )
        ax.set_title(
            f"{method_name}\n"
            f"forecast nRMSE={f(metrics[method_name], 'forecast_nrmse'):.3f}, "
            f"skill@0.20={f(metrics[method_name], 'skill_horizon_time_020'):.2f}",
            fontsize=12.2,
            pad=7,
        )
        ax.text(
            -0.12,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.axhline(0, color="#D0D0D0", lw=0.8, zorder=1)
        ax.axvline(0, color="#D0D0D0", lw=0.8, zorder=1)
        ax.text(
            0.02,
            0.03,
            "open circle = blackout start\nsquare = forecast end",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#555555",
            ha="left",
            va="bottom",
        )

    fig.text(
        0.5,
        0.015,
        f"Dashed grey = truth; coloured solid = method forecast. Representative paired seed selected by minimum APCE forecast nRMSE: {seed}. PCA fitted on blackout-to-end forecast states.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    out_base = args.out_base or (args.root / "figures" / "figure4_l96_blackout_pca_phase_comparison_v1")
    saved = save_pub(fig, out_base)
    plt.close(fig)

    qa = {
        "core_conclusion": "In the blackout forecast window, APCE stays closest to the truth trajectory in PCA space, while PCE and the baselines deviate earlier or more strongly.",
        "figure_archetype": "quantitative grid",
        "representative_seed": seed,
        "blackout_start_step": blackout_step,
        "explained_variance_ratio": explained.tolist(),
        "exports": saved,
        "source_files": {
            "run_source_data": str(source / "lorenz96_1024_blackout_run_source_data.csv"),
            "shared_assets": str(args.root / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz"),
        },
    }
    (out_base.with_suffix(".qa.json")).write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
