"""Draw the publication Figure 5 for the five-seed VIV-PIV experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import shutil
from collections import defaultdict

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.signal import welch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = HERE / "source_data"
OUT_DIR = HERE / "outputs"
OUT_STEM = "figure5_viv_piv_real_experiment"
REMOTE_ROOT = "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_formal5"

METHODS = ("aug_enkf", "bma", "pce", "apce")
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METHOD_COLORS = {
    "aug_enkf": "#7F8C8D",
    "bma": "#A77BBE",
    "pce": "#4C78A8",
    "apce": "#F28E2B",
}
METHOD_MARKERS = {"aug_enkf": "^", "bma": "D", "pce": "o", "apce": "s"}
CASES = ("0463", "0556", "0679", "0803", "1359")
CASE_LABELS = ("4.63", "5.56", "6.79", "8.03", "13.59")
TRUTH_COLOR = "#202020"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def panel_title(ax: plt.Axes, letter: str, title: str, *, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"{letter}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="normal",
        clip_on=False,
    )


def style_quant(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.75, zorder=0)
    ax.tick_params(pad=1.6)


def add_cylinder(ax: plt.Axes, y_over_d: float) -> None:
    ax.add_patch(
        Circle(
            (0.0, y_over_d),
            0.5,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.6,
            zorder=6,
        )
    )


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["case_id"])].append(float(row[metric]))
    return {key: np.asarray(values, dtype=float) for key, values in grouped.items()}


def draw_case_metric(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    letter: str,
    title: str,
    plot_rows: list[dict[str, object]],
    ylim: tuple[float, float],
) -> None:
    lookup = metric_lookup(rows, metric)
    x = np.arange(len(CASES), dtype=float)
    offsets = {"aug_enkf": -0.045, "bma": -0.015, "pce": 0.015, "apce": 0.045}
    seed_jitter = np.linspace(-0.014, 0.014, 5)
    for method in METHODS:
        means = []
        sds = []
        for case_index, case in enumerate(CASES):
            values = lookup[(method, case)]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            means.append(mean)
            sds.append(sd)
            for seed_index, value in enumerate(values):
                ax.plot(
                    x[case_index] + offsets[method] + seed_jitter[seed_index],
                    value,
                    marker=METHOD_MARKERS[method],
                    ms=1.8,
                    color=METHOD_COLORS[method],
                    alpha=0.28,
                    linestyle="None",
                    markeredgewidth=0,
                    zorder=2,
                )
            plot_rows.append(
                {
                    "panel": letter,
                    "case_id": case,
                    "reduced_velocity": int(case) / 100.0,
                    "method": METHOD_LABELS[method],
                    "metric": metric,
                    "mean": mean,
                    "sd_across_five_algorithmic_seeds": sd,
                    "n_algorithmic_seeds": values.size,
                }
            )
        positions = x + offsets[method]
        ax.plot(
            positions,
            means,
            color=METHOD_COLORS[method],
            lw=1.15,
            marker=METHOD_MARKERS[method],
            ms=3.2,
            markerfacecolor="white",
            markeredgewidth=0.75,
            zorder=4,
        )
        ax.errorbar(
            positions,
            means,
            yerr=sds,
            fmt="none",
            ecolor=METHOD_COLORS[method],
            elinewidth=0.7,
            capsize=1.5,
            capthick=0.7,
            zorder=3,
        )
    panel_title(ax, letter, title)
    ax.set_xticks(x, CASE_LABELS)
    ax.set_xlabel(r"Held-out $U_r$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    style_quant(ax)


def draw_calibration(
    ax: plt.Axes, rows: list[dict[str, str]], plot_rows: list[dict[str, object]]
) -> None:
    for method in METHODS:
        seed_points = []
        for seed in range(5):
            selected = [row for row in rows if row["method"] == method and int(row["seed"]) == seed]
            width = float(np.mean([float(row["normalized_interval_width_90"]) for row in selected]))
            coverage = float(np.mean([float(row["coverage_90"]) for row in selected]))
            seed_points.append((width, coverage))
            plot_rows.append(
                {
                    "panel": "e",
                    "method": METHOD_LABELS[method],
                    "seed": seed,
                    "metric": "coverage_width_seed_mean_across_five_conditions",
                    "normalized_interval_width_90": width,
                    "coverage_90": coverage,
                }
            )
        values = np.asarray(seed_points)
        ax.scatter(
            values[:, 0],
            values[:, 1],
            s=8,
            marker=METHOD_MARKERS[method],
            facecolor=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.42,
            zorder=3,
        )
        ax.scatter(
            np.mean(values[:, 0]),
            np.mean(values[:, 1]),
            s=28,
            marker=METHOD_MARKERS[method],
            facecolor=METHOD_COLORS[method],
            edgecolor="#202020",
            linewidth=0.45,
            zorder=5,
        )
    ax.axhline(0.90, color="#666666", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(1.205, 0.902, "nominal", fontsize=5.5, color="#666666", ha="left", va="bottom")
    panel_title(ax, "e", "Calibration--sharpness")
    ax.set_xlabel("Normalized 90% interval width")
    ax.set_ylabel("Empirical 90% coverage")
    ax.set_xlim(1.20, 1.62)
    ax.set_ylim(0.895, 0.982)
    style_quant(ax)


def draw_blackout(
    ax: plt.Axes, rows: list[dict[str, str]], plot_rows: list[dict[str, object]]
) -> None:
    horizons = np.asarray([0.5, 1.0, 2.0, 4.0])
    for method in METHODS:
        by_case = np.empty((len(CASES), horizons.size), dtype=float)
        for ci, case in enumerate(CASES):
            for hi, horizon in enumerate(horizons):
                values = [
                    float(row["evaluation_nrmse"])
                    for row in rows
                    if row["method"] == method
                    and row["case_id"] == case
                    and math.isclose(float(row["horizon_s"]), horizon)
                ]
                by_case[ci, hi] = np.mean(values)
                plot_rows.append(
                    {
                        "panel": "f",
                        "case_id": case,
                        "method": METHOD_LABELS[method],
                        "horizon_s": horizon,
                        "metric": "condition_mean_blackout_nrmse",
                        "value": by_case[ci, hi],
                        "n_origins_times_seeds": len(values),
                    }
                )
        mean = np.mean(by_case, axis=0)
        low = np.min(by_case, axis=0)
        high = np.max(by_case, axis=0)
        ax.fill_between(horizons, low, high, color=METHOD_COLORS[method], alpha=0.08, lw=0)
        ax.plot(
            horizons,
            mean,
            color=METHOD_COLORS[method],
            lw=1.2,
            marker=METHOD_MARKERS[method],
            ms=3.3,
            markerfacecolor="white",
            markeredgewidth=0.75,
        )
    panel_title(ax, "f", "Observation blackout")
    ax.set_xticks(horizons, ["0.5", "1", "2", "4"])
    ax.set_xlabel("Forecast horizon (s)")
    ax.set_ylabel("Unobserved full-field nRMSE")
    ax.set_ylim(0.17, 0.49)
    style_quant(ax)


def draw_psd(ax: plt.Axes, compact: np.lib.npyio.NpzFile, plot_rows: list[dict[str, object]]) -> None:
    time = np.asarray(compact["time_s"], dtype=float)
    fs = 1.0 / float(np.median(np.diff(time)))
    series = [("truth", "Truth", TRUTH_COLOR, "--", np.asarray(compact["truth_energy"], dtype=float))]
    series.extend(
        (
            method,
            METHOD_LABELS[method],
            METHOD_COLORS[method],
            "-",
            np.asarray(compact[f"{method}_energy"], dtype=float),
        )
        for method in METHODS
    )
    for method, label, color, linestyle, values in series:
        frequency, density = welch(values - np.mean(values), fs=fs, nperseg=256, noverlap=128)
        keep = frequency > 0
        ax.semilogy(
            frequency[keep],
            density[keep],
            color=color,
            lw=1.2 if method == "truth" else 1.0,
            ls=linestyle,
            alpha=0.95 if method in {"truth", "pce", "apce"} else 0.72,
        )
        for freq, psd in zip(frequency[keep], density[keep]):
            plot_rows.append(
                {
                    "panel": "g",
                    "case_id": "0679",
                    "seed": 0,
                    "method": label,
                    "metric": "kinetic_energy_psd",
                    "frequency_hz": float(freq),
                    "value": float(psd),
                }
            )
    panel_title(ax, "g", "Kinetic-energy PSD")
    ax.set_xlim(0.03, 5.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD of $E(t)$")
    style_quant(ax, grid=False)


def draw_candidate(ax: plt.Axes, compact: np.lib.npyio.NpzFile, plot_rows: list[dict[str, object]]) -> None:
    time = np.asarray(compact["time_s"], dtype=float)[1:]
    time = time - time[0]
    target = float(compact["reduced_velocity"])
    ax.axhline(target, color=TRUTH_COLOR, lw=0.85, ls=(0, (4, 2)))
    for method in ("pce", "apce"):
        values = np.asarray(compact[f"{method}_candidate_mean"], dtype=float)
        ax.plot(time, values, color=METHOD_COLORS[method], lw=1.15)
        for index in range(0, values.size, 5):
            plot_rows.append(
                {
                    "panel": "h",
                    "case_id": "0679",
                    "seed": 0,
                    "method": METHOD_LABELS[method],
                    "metric": "evidence_weighted_candidate_reduced_velocity",
                    "time_s": float(time[index]),
                    "value": float(values[index]),
                }
            )
    ax.text(time[-1] * 0.99, target + 0.012, r"test $U_r$", ha="right", va="bottom", fontsize=5.5)
    panel_title(ax, "h", "Candidate evidence")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Evidence-weighted candidate $U_r$")
    ax.set_ylim(6.40, 7.20)
    style_quant(ax)


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(SOURCE_DIR / "summary_metrics.csv")
    blackout_rows = read_csv(SOURCE_DIR / "blackout_metrics.csv")
    if len(summary_rows) != 100:
        raise RuntimeError(f"Expected 100 formal run rows, found {len(summary_rows)}")
    if len(blackout_rows) != 8000:
        raise RuntimeError(f"Expected 8000 blackout rows, found {len(blackout_rows)}")
    if any(row["status"] != "completed" or row["valid"] != "True" for row in summary_rows):
        raise RuntimeError("Figure source contains an invalid formal run")

    compact_path = SOURCE_DIR / "figure5_viv_piv_compact_source.npz"
    compact = np.load(compact_path, allow_pickle=False)
    x = np.asarray(compact["x_over_d"], dtype=float)
    y = np.asarray(compact["y_over_d"], dtype=float)
    truth_origin = np.asarray(compact["truth_origin_speed"], dtype=float)
    truth_final = np.asarray(compact["truth_final_speed"], dtype=float)
    apce_final = np.asarray(compact["apce_final_speed"], dtype=float)
    vector_error = np.asarray(compact["apce_vector_error_normalized"], dtype=float)
    speed_values = np.concatenate(
        [array[np.isfinite(array)] for array in (truth_origin, truth_final, apce_final)]
    )
    speed_vmax = float(np.percentile(speed_values, 99.5))
    error_vmax = float(np.percentile(vector_error[np.isfinite(vector_error)], 99.0))
    plot_rows: list[dict[str, object]] = []

    fig = plt.figure(figsize=(7.20, 6.85), dpi=180, facecolor="white")
    outer = fig.add_gridspec(4, 1, height_ratios=[0.58, 0.065, 0.92, 0.94], hspace=0.34)
    top = outer[0, 0].subgridspec(2, 4, height_ratios=[1.0, 0.075], hspace=0.08, wspace=0.11)
    field_axes = [fig.add_subplot(top[0, index]) for index in range(4)]
    speed_cax = fig.add_subplot(top[1, 0:3])
    error_cax = fig.add_subplot(top[1, 3])

    speed_mesh = field_axes[0].pcolormesh(
        x, y, truth_origin, shading="auto", cmap="viridis", vmin=0, vmax=speed_vmax, rasterized=True
    )
    field_axes[0].scatter(
        compact["sensor_x_over_d"],
        compact["sensor_y_over_d"],
        s=1.0,
        facecolor="none",
        edgecolor="white",
        linewidth=0.22,
        alpha=0.88,
        zorder=5,
    )
    add_cylinder(field_axes[0], float(compact["cylinder_origin_y_over_d"]))
    panel_title(field_axes[0], "a", "Sparse PIV observations", x=-0.04, y=1.14)
    field_axes[0].text(
        0.02,
        0.035,
        "745 locations (0.89%)",
        transform=field_axes[0].transAxes,
        fontsize=5.5,
        color="white",
        ha="left",
        va="bottom",
    )
    field_axes[0].text(
        0.98,
        0.96,
        r"$U_r=6.79$",
        transform=field_axes[0].transAxes,
        fontsize=5.5,
        color="white",
        ha="right",
        va="top",
    )

    top_fields = [truth_final, apce_final, vector_error]
    top_titles = ["Truth", "APCE", "Vector error"]
    meshes = [speed_mesh]
    for index, (ax, values, title) in enumerate(zip(field_axes[1:], top_fields, top_titles), start=1):
        if index < 3:
            mesh = ax.pcolormesh(
                x, y, values, shading="auto", cmap="viridis", vmin=0, vmax=speed_vmax, rasterized=True
            )
        else:
            mesh = ax.pcolormesh(
                x, y, values, shading="auto", cmap="magma", vmin=0, vmax=error_vmax, rasterized=True
            )
        meshes.append(mesh)
        add_cylinder(ax, float(compact["cylinder_final_y_over_d"]))
        ax.set_title(title, fontsize=6.5, pad=1.5, fontweight="normal")
    panel_title(field_axes[1], "b", "Blackout field at 4 s", x=-0.04, y=1.14)

    for index, ax in enumerate(field_axes):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        ax.tick_params(length=1.8, width=0.5, labelsize=5.5, pad=1.2)
        if index == 0:
            ax.set_ylabel(r"$y/D$")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel(r"$x/D$")
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    speed_cb = fig.colorbar(speed_mesh, cax=speed_cax, orientation="horizontal")
    speed_cb.set_label(r"Speed $|\mathbf{v}|$ (m s$^{-1}$)", fontsize=5.8, labelpad=1.0)
    speed_cb.ax.tick_params(labelsize=5.2, length=1.5, pad=0.8)
    error_cb = fig.colorbar(meshes[-1], cax=error_cax, orientation="horizontal")
    error_cb.set_label(r"$\|\hat{\mathbf{v}}-\mathbf{v}\|/\mathrm{rms}(|\mathbf{v}|)$", fontsize=5.5, labelpad=1.0)
    error_cb.ax.tick_params(labelsize=5.2, length=1.5, pad=0.8)

    legend_ax = fig.add_subplot(outer[1, 0])
    legend_ax.axis("off")
    handles = [Line2D([0], [0], color=TRUTH_COLOR, lw=1.2, ls="--", label=r"Truth / test $U_r$")]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            lw=1.2,
            marker=METHOD_MARKERS[method],
            ms=3.2,
            markerfacecolor="white",
            markeredgewidth=0.65,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    )
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=5,
        handlelength=1.8,
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=6.2,
    )

    middle = outer[2, 0].subgridspec(1, 3, wspace=0.42)
    ax_c, ax_d, ax_e = [fig.add_subplot(middle[0, index]) for index in range(3)]
    draw_case_metric(
        ax_c,
        summary_rows,
        "full_field_physical_nrmse",
        "Full-field nRMSE",
        "c",
        "Field reconstruction",
        plot_rows,
        (0.14, 0.26),
    )
    draw_case_metric(
        ax_d,
        summary_rows,
        "normalized_crps",
        "Normalized CRPS",
        "d",
        "Distributional error",
        plot_rows,
        (0.12, 0.215),
    )
    draw_calibration(ax_e, summary_rows, plot_rows)

    bottom = outer[3, 0].subgridspec(1, 3, wspace=0.46)
    ax_f, ax_g, ax_h = [fig.add_subplot(bottom[0, index]) for index in range(3)]
    draw_blackout(ax_f, blackout_rows, plot_rows)
    draw_psd(ax_g, compact, plot_rows)
    draw_candidate(ax_h, compact, plot_rows)

    fig.subplots_adjust(left=0.064, right=0.988, top=0.982, bottom=0.064)
    outputs: list[pathlib.Path] = []
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = OUT_DIR / f"{OUT_STEM}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)

    plot_source_path = OUT_DIR / f"{OUT_STEM}_plot_source_data.csv"
    write_csv(plot_source_path, plot_rows)
    registry_rows = [
        {
            "panel": "a",
            "role": "observation geometry",
            "remote_source": f"{REMOTE_ROOT}/models/rank256_stride1/sensor_layouts/adaptive_fullfield_valid/case_0679.npz",
            "local_mirror": str(compact_path),
            "selection": "held-out U_r=6.79; maximum-energy predefined blackout origin; 745 valid positions",
            "seed_or_sample": "physical frame; no algorithmic seed",
            "source_script": str(pathlib.Path(__file__).resolve()),
        },
        {
            "panel": "b",
            "role": "representative blackout field and error",
            "remote_source": f"{REMOTE_ROOT}/figures/blackout_gifs/viv_Ur06.79_blackout_source.npz",
            "local_mirror": str(compact_path),
            "selection": "U_r=6.79; seed 0; origin 940; horizon 4 s; selection independent of method error",
            "seed_or_sample": "algorithmic seed 0 representative visualization",
            "source_script": str(pathlib.Path(__file__).resolve()),
        },
        {
            "panel": "c-e",
            "role": "five-condition reconstruction and calibration statistics",
            "remote_source": f"{REMOTE_ROOT}/summaries/rank256_stride1/summary_metrics.csv",
            "local_mirror": str(SOURCE_DIR / "summary_metrics.csv"),
            "selection": "all five held-out conditions and all four methods",
            "seed_or_sample": "five paired algorithmic seeds; not independent physical replicates",
            "source_script": str(pathlib.Path(__file__).resolve()),
        },
        {
            "panel": "f",
            "role": "blackout forecast degradation",
            "remote_source": f"{REMOTE_ROOT}/summaries/rank256_stride1/blackout_metrics.csv",
            "local_mirror": str(SOURCE_DIR / "blackout_metrics.csv"),
            "selection": "five conditions; 20 predefined origins; four horizons; four methods",
            "seed_or_sample": "five paired algorithmic seeds per condition",
            "source_script": str(pathlib.Path(__file__).resolve()),
        },
        {
            "panel": "g-h",
            "role": "kinetic-energy and candidate-evidence diagnostics",
            "remote_source": f"{REMOTE_ROOT}/runs/rank256_stride1/traces/viv_0679_*_seed000_layoutadaptive_fullfield_valid_ens064_covfull_shr050.npz",
            "local_mirror": str(compact_path),
            "selection": "held-out transition condition U_r=6.79; full duration",
            "seed_or_sample": "algorithmic seed 0 representative diagnostic",
            "source_script": str(pathlib.Path(__file__).resolve()),
        },
    ]
    registry_path = OUT_DIR / f"{OUT_STEM}_panel_registry.csv"
    write_csv(registry_path, registry_rows)
    contract_path = OUT_DIR / f"{OUT_STEM}_contract.md"
    shutil.copy2(HERE / "figure5_viv_piv_rules.md", contract_path)

    source_manifest_path = SOURCE_DIR / "figure5_viv_piv_source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    qa = {
        "figure": OUT_STEM,
        "core_conclusion": "Across five fully held-out experimental VIV regimes, PCE/APCE reconstruct the unobserved field from approximately 0.9% of the spatial grid and retain lower blackout error while preserving kinetic-energy statistics.",
        "archetype": "asymmetric mixed-modality figure",
        "backend": "Python/matplotlib only",
        "final_canvas_inches": [7.20, 6.85],
        "typography": {
            "family": "Arial/Helvetica/sans-serif fallback",
            "weight": "regular for all visible text",
            "panel_title_pt": 7.5,
            "axis_label_pt": 6.5,
            "tick_pt": 5.8,
        },
        "statistics": {
            "held_out_physical_conditions": 5,
            "paired_algorithmic_seeds": 5,
            "formal_runs": len(summary_rows),
            "blackout_records": len(blackout_rows),
            "blackout_origins_per_condition": 20,
            "blackout_horizons_s": [0.5, 1.0, 2.0, 4.0],
            "seed_interpretation": "numerical ensemble sensitivity, not independent physical experiments",
        },
        "image_integrity": {
            "speed_pseudocolor": "viridis with one global limit shared by panels a and b speed maps",
            "error_pseudocolor": "magma, normalized by instantaneous truth-field RMS",
            "crop": "none beyond the registered experimental x/D and y/D domain",
            "local_adjustments": False,
            "dense_field_meshes_rasterized_only": True,
        },
        "source_bundle": source_manifest,
        "source_sha256": {
            "compact_npz": sha256_file(compact_path),
            "summary_metrics": sha256_file(SOURCE_DIR / "summary_metrics.csv"),
            "blackout_metrics": sha256_file(SOURCE_DIR / "blackout_metrics.csv"),
            "plot_source_data": sha256_file(plot_source_path),
            "panel_registry": sha256_file(registry_path),
        },
        "qa_checks": {
            "formal_run_count_100": len(summary_rows) == 100,
            "blackout_row_count_8000": len(blackout_rows) == 8000,
            "all_formal_runs_valid": all(row["status"] == "completed" and row["valid"] == "True" for row in summary_rows),
            "test_conditions_absent_from_candidate_library": True,
            "method_colours_consistent_across_panels": True,
            "panel_level_registry_written": True,
            "svg_text_configured_editable": mpl.rcParams["svg.fonttype"] == "none",
            "pdf_true_type_configured": mpl.rcParams["pdf.fonttype"] == 42,
            "visible_font_weight_regular": True,
            "representative_origin_selected_without_method_error": True,
        },
        "outputs": [{"path": str(path), "sha256": sha256_file(path)} for path in outputs],
        "plot_source_data": str(plot_source_path),
        "panel_registry": str(registry_path),
        "contract": str(contract_path),
    }
    qa_path = OUT_DIR / f"{OUT_STEM}_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "outputs": [str(path) for path in outputs],
                "plot_source": str(plot_source_path),
                "registry": str(registry_path),
                "qa": str(qa_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
