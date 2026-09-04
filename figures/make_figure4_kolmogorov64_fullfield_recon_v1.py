from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "source_data" / "figure4_kolmogorov64_re575_k4_obs8x8_t2_seed2026081600"
RUN_CSV = SOURCE_ROOT / "kolmogorov64_run_source_data.csv"
SUMMARY_CSV = SOURCE_ROOT / "kolmogorov64_method_summary.csv"
DEFAULT_OUTPUT = HERE / "figure4_kolmogorov64_fullfield_recon_v1"

PANEL_TITLES = ["Observations", "Truth", "Aug-EnKF", "BMA", "APCE"]
METHOD_KEYS = ["aug_enkf", "bma_static", "apce"]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
        }
    )


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02)


def field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kol_field",
        ["#204a78", "#f3f1ea", "#a2443a"],
        N=256,
    )


def format_tick(x: float) -> str:
    text = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", "+0", ""} else text


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_seed(row: dict[str, str]) -> int:
    return int(float(row["seed"]))


def row_metric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    return float(value)


def select_best_apce_row(rows: list[dict[str, str]], *, sensor_grid: int, obs_interval: int) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row.get("case") == "kolmogorov64"
        and row.get("method") == "apce"
        and int(float(row["sensor_grid"])) == sensor_grid
        and int(float(row["obs_interval"])) == obs_interval
        and row.get("status") == "completed"
        and row.get("numerical_status") == "valid"
    ]
    if not candidates:
        raise RuntimeError("No valid APCE rows found for the requested KOL configuration.")
    return min(candidates, key=lambda row: row_metric(row, "nrmse"))


def find_row(
    rows: list[dict[str, str]],
    *,
    method: str,
    seed: int,
    sensor_grid: int,
    obs_interval: int,
) -> dict[str, str]:
    for row in rows:
        if (
            row.get("case") == "kolmogorov64"
            and row.get("method") == method
            and int(float(row["seed"])) == seed
            and int(float(row["sensor_grid"])) == sensor_grid
            and int(float(row["obs_interval"])) == obs_interval
            and row.get("status") == "completed"
            and row.get("numerical_status") == "valid"
    ):
            return row
    raise RuntimeError(f"Missing valid row for method={method}, seed={seed}.")


def resolve_trace_path(row: dict[str, str], *, source_root: Path, method_key: str, seed: int) -> Path:
    local = source_root / "traces" / f"{method_key}_seed_{seed}.npz"
    if local.exists():
        return local
    remote = Path(row["trace_npz"])
    return remote


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def sparse_observation_map(observations: np.ndarray, observation_indices: np.ndarray, *, nx: int, ny: int, frame: int) -> np.ndarray:
    flat = np.full(nx * ny, np.nan, dtype=float)
    flat[np.asarray(observation_indices, dtype=int)] = np.asarray(observations[frame], dtype=float)
    return flat.reshape(ny, nx)


def reshape_field(flat: np.ndarray, *, nx: int, ny: int) -> np.ndarray:
    return np.asarray(flat, dtype=float).reshape(ny, nx)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#222222",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a full-field KOL reconstruction comparison figure.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=None, help="Preferred seed. If omitted, best APCE seed is selected automatically.")
    parser.add_argument("--sensor-grid", type=int, default=8)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--frame-index", type=int, default=100)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--run-csv", type=Path, default=RUN_CSV)
    args = parser.parse_args()

    configure_matplotlib()

    rows = load_rows(args.run_csv)
    if args.seed is None:
        selected_apce = select_best_apce_row(rows, sensor_grid=args.sensor_grid, obs_interval=args.obs_interval)
        seed = row_seed(selected_apce)
    else:
        seed = int(args.seed)
        selected_apce = find_row(rows, method="apce", seed=seed, sensor_grid=args.sensor_grid, obs_interval=args.obs_interval)

    method_rows: dict[str, dict[str, str]] = {}
    for method_key in METHOD_KEYS:
        method_rows[method_key] = find_row(rows, method=method_key, seed=seed, sensor_grid=args.sensor_grid, obs_interval=args.obs_interval)

    traces: dict[str, dict[str, np.ndarray]] = {}
    for method_key in METHOD_KEYS:
        trace_path = resolve_trace_path(method_rows[method_key], source_root=args.source_root, method_key=method_key, seed=seed)
        traces[method_key] = load_trace(trace_path)

    apce_trace = traces["apce"]
    truth = reshape_field(apce_trace["truth_states"][args.frame_index], nx=int(apce_trace["nx"]), ny=int(apce_trace["ny"]))
    obs_map = sparse_observation_map(
        apce_trace["observations"],
        apce_trace["observation_indices"],
        nx=int(apce_trace["nx"]),
        ny=int(apce_trace["ny"]),
        frame=args.frame_index,
    )
    aug_enkf = reshape_field(traces["aug_enkf"]["mean_states"][args.frame_index], nx=int(apce_trace["nx"]), ny=int(apce_trace["ny"]))
    bma = reshape_field(traces["bma_static"]["mean_states"][args.frame_index], nx=int(apce_trace["nx"]), ny=int(apce_trace["ny"]))
    apce = reshape_field(traces["apce"]["mean_states"][args.frame_index], nx=int(apce_trace["nx"]), ny=int(apce_trace["ny"]))

    combined = np.concatenate(
        [
            truth[np.isfinite(truth)].ravel(),
            aug_enkf[np.isfinite(aug_enkf)].ravel(),
            bma[np.isfinite(bma)].ravel(),
            apce[np.isfinite(apce)].ravel(),
            obs_map[np.isfinite(obs_map)].ravel(),
        ]
    )
    vmax = float(np.nanpercentile(np.abs(combined), 99.3))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(combined)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    cmap = field_cmap()
    cmap.set_bad("white", alpha=1.0)

    fig = plt.figure(figsize=(12.6, 3.25))
    gs = fig.add_gridspec(
        1,
        5,
        left=0.03,
        right=0.995,
        top=0.84,
        bottom=0.22,
        wspace=0.11,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]

    ims = []
    obs_y, obs_x = np.nonzero(np.isfinite(obs_map))
    ims.append(
        axes[0].scatter(
            obs_x,
            obs_y,
            c=obs_map[np.isfinite(obs_map)],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            marker="s",
            s=26,
            linewidths=0,
            rasterized=True,
        )
    )
    axes[0].set_xlim(-0.5, obs_map.shape[1] - 0.5)
    axes[0].set_ylim(-0.5, obs_map.shape[0] - 0.5)
    axes[0].set_aspect("equal")
    ims.append(
        axes[1].imshow(
            truth,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            rasterized=True,
        )
    )
    ims.append(
        axes[2].imshow(
            aug_enkf,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            rasterized=True,
        )
    )
    ims.append(
        axes[3].imshow(
            bma,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            rasterized=True,
        )
    )
    ims.append(
        axes[4].imshow(
            apce,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            rasterized=True,
        )
    )

    for idx, (ax, title) in enumerate(zip(axes, PANEL_TITLES, strict=True)):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=13.0, pad=5.0)
        add_panel_label(ax, chr(ord("a") + idx))

    # Keep the visual story honest: the figure should not overclaim a success case.
    seed_label = f"seed {seed}"
    fig.text(0.03, 0.965, "KOL full-field reconstruction", ha="left", va="top", fontsize=13.5)
    fig.text(
        0.995,
        0.965,
        f"Re=575, k=4, 8×8 obs, t=2, {seed_label}",
        ha="right",
        va="top",
        fontsize=9.2,
        color="#555555",
    )

    cax = fig.add_axes([0.23, 0.12, 0.54, 0.038])
    cb = fig.colorbar(ims[-1], cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    ticks = np.linspace(-vmax, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([format_tick(t) for t in ticks])
    cb.ax.tick_params(labelsize=8.0, length=0, pad=1.0)
    cax.text(0.5, -1.25, "vorticity", transform=cax.transAxes, ha="center", va="top", fontsize=8.5)

    save_all(fig, args.output)

    summary_rows: list[dict[str, str]] = []
    if args.summary_csv.exists():
        with args.summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
    metrics = {}
    for method_key in ["aug_enkf", "bma_static", "apce"]:
        row = method_rows[method_key]
        metrics[method_key] = {
            "nrmse": row_metric(row, "nrmse"),
            "crps": row_metric(row, "crps"),
            "alpha_mae": row_metric(row, "alpha_absolute_error"),
            "coverage_90": row_metric(row, "coverage_90"),
        }

    qa = {
        "source_root": str(args.source_root),
        "run_csv": str(args.run_csv),
        "summary_csv": str(args.summary_csv),
        "selected_seed": seed,
        "selected_frame_index": int(args.frame_index),
        "selected_apce_row_json": selected_apce.get("_json_path", ""),
        "methods": {
            "aug_enkf": str(resolve_trace_path(method_rows["aug_enkf"], source_root=args.source_root, method_key="aug_enkf", seed=seed)),
            "bma_static": str(resolve_trace_path(method_rows["bma_static"], source_root=args.source_root, method_key="bma_static", seed=seed)),
            "apce": str(resolve_trace_path(method_rows["apce"], source_root=args.source_root, method_key="apce", seed=seed)),
        },
        "truth_shape": list(truth.shape),
        "observations_shape": list(obs_map.shape),
        "vmax": vmax,
        "metrics": metrics,
        "note": "Full-field comparison plate. The reconstruction remains visibly imperfect; the figure is intended to diagnose boundary behavior rather than overstate success.",
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
