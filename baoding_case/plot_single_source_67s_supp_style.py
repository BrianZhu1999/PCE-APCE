#!/usr/bin/env python3
"""Draw the Baoding single-source row for Supplementary Data Figure 2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


START_S, END_S = 46254.0, 46320.0
FIGURE_DPI = 650
FIGURE_WIDTH_PX = 11532
FIG_W = FIGURE_WIDTH_PX / FIGURE_DPI
FIG_H = 4.70

# Figure 4 master typography, inherited by Figure 5.
FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11

GPS_COLOR = "#202020"
PCE_COLOR = "#4C78A8"
APCE_COLOR = "#F28E2B"
TEXT_COLOR = "#111111"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    positions = np.asarray(
        [[float(row[key]) for key in ("px", "py", "pz")] for row in rows],
        dtype=float,
    )
    order = np.argsort(times)
    return times[order], positions[order]


def load_method_runs(root: Path, method: str):
    paths = sorted((root / "runs").glob(f"{method}_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five {method} runs, found {len(paths)}")

    all_positions: list[np.ndarray] = []
    all_widths: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    reference_times: np.ndarray | None = None
    seeds: list[int] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [
            row
            for row in payload["records"]
            if START_S <= float(row["time_s"]) <= END_S
        ]
        times = np.asarray([float(row["time_s"]) for row in records], dtype=float)
        if reference_times is None:
            reference_times = times
        elif times.shape != reference_times.shape or not np.allclose(times, reference_times):
            raise RuntimeError(f"inconsistent time grid in {path}")
        all_positions.append(
            np.asarray(
                [[float(row[key]) for key in ("px", "py", "pz")] for row in records],
                dtype=float,
            )
        )
        all_widths.append(
            np.asarray([float(row["interval_width_m"]) for row in records], dtype=float)
        )
        all_errors.append(
            np.asarray([float(row["position_error_m"]) for row in records], dtype=float)
        )
        seeds.append(int(payload["seed"]))

    assert reference_times is not None
    positions = np.asarray(all_positions)
    widths = np.asarray(all_widths)
    errors = np.asarray(all_errors)
    return {
        "times": reference_times,
        "median_position": np.median(positions, axis=0),
        "median_interval_width": np.median(widths, axis=0),
        "positions": positions,
        "errors": errors,
        "paths": paths,
        "seeds": seeds,
    }


def interpolate_truth(
    sample_times: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
) -> np.ndarray:
    if sample_times[0] < truth_times[0] or sample_times[-1] > truth_times[-1]:
        raise RuntimeError("GPS truth does not cover the selected method time grid")
    return np.column_stack(
        [np.interp(sample_times, truth_times, truth_positions[:, dim]) for dim in range(3)]
    )


def trajectory_rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((estimates - truth) ** 2, axis=1))))


def pooled_position_rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def ribbon_polygon(xy: np.ndarray, radius: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(xy)
    tangent[0] = xy[1] - xy[0]
    tangent[-1] = xy[-1] - xy[-2]
    tangent[1:-1] = xy[2:] - xy[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = xy + normal * radius[:, None]
    right = xy - normal * radius[:, None]
    return np.vstack((left, right[::-1]))


def add_panel_header(fig: plt.Figure, left_in: float, letter: str, title: str) -> None:
    baseline = 4.25 / FIG_H
    fig.text(
        left_in / FIG_W,
        baseline,
        letter,
        fontsize=FONT_PANEL,
        fontweight="bold",
        ha="left",
        va="baseline",
        color=TEXT_COLOR,
    )
    fig.text(
        (left_in + 0.46) / FIG_W,
        baseline,
        title,
        fontsize=FONT_TITLE,
        fontweight="normal",
        ha="left",
        va="baseline",
        color=TEXT_COLOR,
    )


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.labelsize": FONT_AXIS,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def style_3d_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK, pad=1, width=0.9)
    ax.zaxis.set_tick_params(labelsize=FONT_TICK, pad=1, width=0.9)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#A8A8A8")
        axis._axinfo["grid"].update(color=(0.84, 0.84, 0.84, 0.72), linewidth=0.55)


def save_outputs(fig: plt.Figure, stem: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    settings = {
        "png": {"dpi": FIGURE_DPI},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }
    for extension, kwargs in settings.items():
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[extension] = str(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--pce-runs", type=Path, required=True)
    parser.add_argument("--apce-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configure_matplotlib()
    truth_times, truth_positions = load_truth(args.frontend / "gps_truth.csv")
    pce_run = load_method_runs(args.pce_runs, "pce")
    apce_run = load_method_runs(args.apce_runs, "apce")
    times = pce_run["times"]
    if times.shape != apce_run["times"].shape or not np.allclose(times, apce_run["times"]):
        raise RuntimeError("PCE and APCE time grids differ")
    if len(times) != 67:
        raise RuntimeError(f"expected 67 one-second frames, found {len(times)}")

    truth = interpolate_truth(times, truth_times, truth_positions)
    pce = pce_run["median_position"]
    apce = apce_run["median_position"]
    origin = np.mean(truth, axis=0)
    truth_o, pce_o, apce_o = truth - origin, pce - origin, apce - origin

    # The run files retain the mean marginal 90% interval width, rather than
    # the full EN covariance. Half-width is therefore shown as an isotropic
    # normal ribbon and is registered explicitly as a display proxy.
    apce_radius = apce_run["median_interval_width"] / 2.0
    ribbon = ribbon_polygon(apce_o[:, :2], apce_radius)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax3 = fig.add_axes(
        [0.42 / FIG_W, 0.67 / FIG_H, 9.35 / FIG_W, 3.36 / FIG_H], projection="3d"
    )
    ax2 = fig.add_axes([10.25 / FIG_W, 0.67 / FIG_H, 7.05 / FIG_W, 3.36 / FIG_H])

    trajectory_styles = (
        (truth_o, GPS_COLOR, "GPS", 1.45, "--", None, None),
        (pce_o, PCE_COLOR, "PCE", 1.30, "-", "o", (1, 8)),
        (apce_o, APCE_COLOR, "APCE", 1.40, "-", "s", (4, 8)),
    )
    for data, color, label, width, linestyle, marker, markevery in trajectory_styles:
        ax3.plot(
            data[:, 0], data[:, 1], data[:, 2],
            color=color, lw=width, ls=linestyle, marker=marker, markevery=markevery,
            ms=3.5, markerfacecolor="white", markeredgewidth=0.75,
            label=label, zorder=5 if label == "APCE" else 4,
        )
    ax3.set_xlabel("East offset (m)", labelpad=3)
    ax3.set_ylabel("North offset (m)", labelpad=3)
    ax3.set_zlabel("Up offset (m)", labelpad=2)
    ax3.view_init(elev=25, azim=-58)
    combined = np.concatenate((truth_o, pce_o, apce_o), axis=0)
    span = np.maximum(np.ptp(combined, axis=0), 1.0)
    ax3.set_box_aspect((span[0], span[1], 5.0 * span[2]))
    style_3d_axes(ax3)

    ax2.fill(
        ribbon[:, 0], ribbon[:, 1], color=APCE_COLOR, alpha=0.18,
        linewidth=0, zorder=1,
    )
    for data, color, label, width, linestyle, marker, markevery in trajectory_styles:
        ax2.plot(
            data[:, 0], data[:, 1], color=color, lw=width, ls=linestyle,
            marker=marker, markevery=markevery, ms=3.5,
            markerfacecolor="white", markeredgewidth=0.75,
            label=label, zorder=5 if label == "APCE" else 4,
        )
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("East offset (m)")
    ax2.set_ylabel("North offset (m)")
    ax2.tick_params(axis="both", which="major", labelsize=FONT_TICK, pad=2, width=0.9)
    ax2.grid(axis="both", color="#E2E2E2", lw=0.5, alpha=0.72, zorder=0)

    add_panel_header(fig, 0.18, "a", "Three-dimensional trajectory")
    add_panel_header(fig, 9.97, "b", "Horizontal trajectory and uncertainty")

    legend_handles = [
        Line2D([0], [0], color=GPS_COLOR, lw=1.45, ls="--", label="GPS"),
        Line2D([0], [0], color=PCE_COLOR, lw=1.30, marker="o", ms=4.2, markerfacecolor="white", label="PCE"),
        Line2D([0], [0], color=APCE_COLOR, lw=1.40, marker="s", ms=4.2, markerfacecolor="white", label="APCE"),
        Patch(facecolor=APCE_COLOR, edgecolor="none", alpha=0.18, label="APCE 90% width"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=4, fontsize=FONT_LEGEND, handlelength=1.7,
        columnspacing=1.15, borderaxespad=0.0,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "supplementary_data_figure2_single_source_67s_q2_12_scale1"
    outputs = save_outputs(fig, stem)
    plt.close(fig)

    source_rows = []
    for idx, time_s in enumerate(times):
        source_rows.append(
            {
                "time_s": time_s,
                "gps_E": truth[idx, 0], "gps_N": truth[idx, 1], "gps_U": truth[idx, 2],
                "pce_E_median_across_5_seeds": pce[idx, 0],
                "pce_N_median_across_5_seeds": pce[idx, 1],
                "pce_U_median_across_5_seeds": pce[idx, 2],
                "apce_E_median_across_5_seeds": apce[idx, 0],
                "apce_N_median_across_5_seeds": apce[idx, 1],
                "apce_U_median_across_5_seeds": apce[idx, 2],
                "pce_interval_width_m_median_across_5_seeds": pce_run["median_interval_width"][idx],
                "apce_interval_width_m_median_across_5_seeds": apce_run["median_interval_width"][idx],
            }
        )
    source_path = args.output / "supplementary_data_figure2_single_source_67s_source.csv"
    with source_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    panel_registry_path = args.output / "supplementary_data_figure2_single_source_67s_panel_registry.csv"
    panel_rows = [
        {
            "panel": "a", "role": "3D trajectory geometry",
            "selection": "preselected 12:50:54--12:52:00 near-full-circle window",
            "display": "GPS and five-seed median PCE/APCE trajectories; vertical scale exaggerated 5x",
            "frontend": str(args.frontend), "pce_runs": str(args.pce_runs), "apce_runs": str(args.apce_runs),
        },
        {
            "panel": "b", "role": "horizontal trajectory and state-uncertainty display",
            "selection": "same fixed 67-frame window as panel a",
            "display": "APCE mean-marginal 90% interval half-width rendered as isotropic normal ribbon proxy",
            "frontend": str(args.frontend), "pce_runs": str(args.pce_runs), "apce_runs": str(args.apce_runs),
        },
    ]
    with panel_registry_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel_rows[0]))
        writer.writeheader()
        writer.writerows(panel_rows)

    script_path = Path(__file__).resolve()
    registry = {
        "figure_contract": {
            "core_conclusion": "PCE and APCE retain the geometry of a preselected near-full-circle single-source flight segment, with APCE providing the lower pooled position RMSE.",
            "evidence_chain": {
                "a": "three-dimensional trajectory geometry with registered 5x vertical display exaggeration",
                "b": "horizontal trajectory agreement and the retained APCE state-interval width",
            },
            "archetype": "asymmetric mixed-modality figure row",
            "backend": "Python/matplotlib only",
        },
        "typography": {
            "reference": "main-text Figure 4 master rules inherited by Figure 5",
            "panel_label_pt": FONT_PANEL, "panel_title_pt": FONT_TITLE,
            "legend_pt": FONT_LEGEND, "axis_label_pt": FONT_AXIS, "tick_pt": FONT_TICK,
            "only_panel_letters_bold": True, "sentence_annotation_present": False,
        },
        "window": {
            "start_time_s": START_S, "end_time_s": END_S, "frames": len(times),
            "update_interval_s": float(np.median(np.diff(times))),
            "selection_status": "fixed before this figure redesign",
        },
        "configuration": {
            "q_min_accel_mps2": 2.0, "q_max_accel_mps2": 12.0,
            "observation_covariance_scale": 1.0, "turn_rate_radps": -0.10,
            "ensemble_members": 48, "seeds": pce_run["seeds"],
        },
        "metrics": {
            "pce_pooled_five_seed_position_rmse_m": pooled_position_rmse(pce_run["errors"]),
            "apce_pooled_five_seed_position_rmse_m": pooled_position_rmse(apce_run["errors"]),
            "pce_plotted_median_trajectory_rmse_m": trajectory_rmse(pce, truth),
            "apce_plotted_median_trajectory_rmse_m": trajectory_rmse(apce, truth),
        },
        "display": {
            "master_width_px_at_650_dpi": FIGURE_WIDTH_PX, "vertical_exaggeration": 5.0,
            "uncertainty": "APCE scalar mean marginal 90% interval width divided by two and rendered as an isotropic 2D normal ribbon proxy; no full EN covariance is present in the run records; no 3D tube is drawn",
        },
        "gps_role": "offline scoring and display only; GPS is not an assimilation input",
        "sources": {
            "frontend": str(args.frontend),
            "pce_runs": [str(path) for path in pce_run["paths"]],
            "apce_runs": [str(path) for path in apce_run["paths"]],
        },
        "outputs": outputs, "source_csv": str(source_path),
        "panel_registry": str(panel_registry_path), "script": str(script_path),
        "script_sha256": sha256_file(script_path),
    }
    registry_path = args.output / "supplementary_data_figure2_single_source_67s_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**registry, "registry": str(registry_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
