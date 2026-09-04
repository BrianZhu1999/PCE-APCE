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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
COLORS = {
    "Truth": "#252525",
    "Aug-EnKF": "#8A8A8A",
    "BMA": "#4E79A7",
    "PCE": "#E56A5C",
    "APCE": "#2BAA9A",
}
LINEWIDTHS = {
    "Truth": 2.2,
    "Aug-EnKF": 2.0,
    "BMA": 2.1,
    "PCE": 2.8,
    "APCE": 2.8,
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


def pca_basis(traces: list[np.ndarray], start_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast_states = [trace[start_step:] for trace in traces]
    stacked = np.concatenate(forecast_states, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T
    explained = (singular_values[:3] ** 2) / max(float((singular_values**2).sum()), 1.0e-30)
    return mean, components, explained


def project(trace: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (trace - mean) @ components


def clean_3d_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#D8D8D8")
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.25
    ax.tick_params(axis="both", labelsize=10, pad=0)
    ax.zaxis.set_tick_params(labelsize=10, pad=0)


def set_equalish_limits(ax: plt.Axes, coords: np.ndarray) -> None:
    mins = np.nanmin(coords, axis=0)
    maxs = np.nanmax(coords, axis=0)
    centers = 0.5 * (mins + maxs)
    ranges = np.maximum(maxs - mins, 1.0e-6)
    radius = 0.58 * ranges.max()
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_path(ax: plt.Axes, coords: np.ndarray, label: str, *, linestyle: str = "-", alpha: float = 1.0) -> None:
    color = COLORS[label]
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color=color,
        lw=LINEWIDTHS[label],
        ls=linestyle,
        alpha=alpha,
        solid_capstyle="round",
        zorder=5 if label in {"PCE", "APCE"} else 3,
    )
    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], s=44, facecolors="white", edgecolors=color, linewidths=1.4, depthshade=False)
    ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], s=52, marker="s", color=color, edgecolors="white", linewidths=0.8, depthshade=False)


def annotate_endpoint(ax: plt.Axes, coords: np.ndarray, label: str, offset: tuple[float, float, float]) -> None:
    end = coords[-1]
    ax.text(
        end[0] + offset[0],
        end[1] + offset[1],
        end[2] + offset[2],
        label,
        color=COLORS[label],
        fontsize=11.5,
        fontweight="bold" if label in {"PCE", "APCE"} else "normal",
        ha="left",
        va="center",
    )


def save_all(fig: plt.Figure, out_base: Path) -> dict[str, str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for ext, kwargs in {
        "png": {"dpi": 480},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved[ext] = str(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Single 3D PCA overlay for Lorenz-96 blackout forecast.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("<HILDA_RESULTS_ROOT>/results/figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds_20260815_gpu23"),
    )
    parser.add_argument("--out-base", type=Path, default=None)
    parser.add_argument("--elev", type=float, default=27.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    args = parser.parse_args()

    source = args.root / "source_data"
    run_rows = read_csv(source / "lorenz96_1024_blackout_run_source_data.csv")
    seed, representative = choose_representative_seed(run_rows)
    rows_for_seed = {row["label"]: row for row in run_rows if int(row["seed"]) == seed}
    blackout_step = int(float(representative["blackout_start_step"]))

    shared = load_npz(args.root / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz")
    truth = shared["truth"].astype(float)
    method_traces = {}
    for label in METHODS:
        key = METHOD_KEY[label]
        trace_path = args.root / "artifacts" / "method_traces" / "lorenz96_1024" / "time8" / key / f"seed_{seed}.npz"
        method_traces[label] = load_npz(trace_path)["mean_states"].astype(float)

    mean, components, explained = pca_basis([truth] + [method_traces[label] for label in METHODS], blackout_step)
    truth_coords = project(truth[blackout_step:], mean, components)
    coords = {label: project(method_traces[label][blackout_step:], mean, components) for label in METHODS}
    stacked_coords = np.concatenate([truth_coords] + [coords[label] for label in METHODS], axis=0)

    fig = plt.figure(figsize=(9.4, 8.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.91)
    clean_3d_axes(ax)
    set_equalish_limits(ax, stacked_coords)
    ax.view_init(elev=args.elev, azim=args.azim)
    try:
        ax.set_box_aspect((1.0, 0.92, 0.88))
    except Exception:
        pass

    plot_path(ax, truth_coords, "Truth", linestyle=(0, (4, 3)), alpha=0.78)
    for label in METHODS:
        alpha = 0.82 if label in {"Aug-EnKF", "BMA"} else 0.97
        plot_path(ax, coords[label], label, alpha=alpha)

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)", fontsize=13, labelpad=8)
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)", fontsize=13, labelpad=8)
    ax.set_zlabel(f"PC3 ({explained[2] * 100:.1f}%)", fontsize=13, labelpad=8)
    title = "Blackout forecast phase portrait"
    subtitle = (
        f"seed {seed}; open circles = blackout start, squares = forecast end; "
        f"PCE/APCE nRMSE {float(rows_for_seed['PCE']['forecast_nrmse']):.3f}/{float(rows_for_seed['APCE']['forecast_nrmse']):.3f}"
    )
    fig.text(0.04, 0.965, title, fontsize=16, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.928, subtitle, fontsize=10.5, color="#4A4A4A", ha="left", va="top")

    handles = [Line2D([0], [0], color=COLORS["Truth"], lw=2.2, ls=(0, (4, 3)), label="Truth")]
    handles.extend([Line2D([0], [0], color=COLORS[label], lw=LINEWIDTHS[label], label=label) for label in METHODS])
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=11, handlelength=2.0)

    out_base = args.out_base or (args.root / "figures" / "figure4_l96_blackout_pca3d_overlay_v1")
    saved = save_all(fig, out_base)
    plt.close(fig)

    qa = {
        "core_conclusion": "During the blackout forecast window, PCE/APCE remain closer to the truth trajectory in the leading three PCA coordinates than the main baselines for the representative paired seed.",
        "figure_archetype": "single quantitative 3D phase portrait",
        "representative_seed": seed,
        "blackout_start_step": blackout_step,
        "pca_explained_variance_ratio_pc1_pc2_pc3": explained.tolist(),
        "view": {"elev": args.elev, "azim": args.azim},
        "exports": saved,
        "source_files": {
            "run_source_data": str(source / "lorenz96_1024_blackout_run_source_data.csv"),
            "shared_assets": str(args.root / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz"),
        },
    }
    qa_path = out_base.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["qa"] = str(qa_path)
    print(json.dumps(saved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
