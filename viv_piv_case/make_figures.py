"""Python-only, publication-oriented figures for the VIV-PIV case.

Figure contract: the figures support one bounded claim: shadow-anchored
predictive evidence can be audited alongside held-out wake reconstruction and
known-input blackout forecasts.  The panels therefore show both deterministic
field/energy errors and probabilistic calibration; candidate weights are
labelled operational evidence rather than posterior parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

import matplotlib as mpl
import numpy as np

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend selected before pyplot import)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.frameon": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

COLORS = {"pce": "#2f6f9f", "apce": "#c65a3a", "truth": "#222222", "train": "#6b8e9f", "test": "#c65a3a"}


def save_pub(fig: mpl.figure.Figure, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_trace(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def preferred_trace(run_root: pathlib.Path, case_id: str, method: str) -> pathlib.Path | None:
    """Select the final full-covariance ensemble-64 trace for a case/method."""
    candidates = [
        path for path in (run_root / "traces").glob(f"viv_{case_id}_{method}_seed000*.npz")
        if "layout20x40_ens064_covfull" in path.name and "covtaper" not in path.name
    ]
    if not candidates:
        return None
    # Prefer the explicit shr050 dynamic run; APCE rank-256 uses an audited
    # shr050 archive and therefore has no suffix.
    return sorted(candidates, key=lambda path: ("shr050" in path.name, path.name), reverse=True)[0]


def run_payloads(run_root: pathlib.Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in (run_root / "runs").glob("viv_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[(str(payload.get("case_id")), str(payload.get("method")), int(payload.get("seed", 0)))] = payload
    return output


def figure_split_pod_candidates(config: dict[str, Any], model_root: pathlib.Path, out: pathlib.Path) -> None:
    manifest = json.loads((model_root / "model_manifest.json").read_text(encoding="utf-8"))
    pod = PODModel.load(model_root / "pod_model.npz")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw={"wspace": 0.34})
    train = np.asarray([int(x) / 100 for x in config["train_cases"]])
    test = np.asarray([int(x) / 100 for x in config["test_cases"]])
    axes[0].scatter(train, np.zeros_like(train), s=24, c=COLORS["train"], label="training candidates", zorder=3)
    axes[0].scatter(test, np.ones_like(test), s=28, c=COLORS["test"], marker="D", label="external tests", zorder=3)
    axes[0].set_yticks([0, 1], ["train", "test"])
    axes[0].set_xlabel(r"reduced velocity $U_r$")
    axes[0].set_title("External split")
    axes[0].legend(loc="center right", fontsize=6)
    cumulative = pod.explained_fraction[: min(256, pod.explained_fraction.size)] * 100
    axes[1].plot(np.arange(1, cumulative.size + 1), cumulative, color=COLORS["pce"], lw=1.3)
    axes[1].axvline(pod.rank, color=COLORS["apce"], ls="--", lw=1.0)
    axes[1].axhline(cumulative[pod.rank - 1], color="#888888", ls=":", lw=0.8)
    axes[1].set_xlabel("POD rank")
    axes[1].set_ylabel("cumulative energy (%)")
    axes[1].set_title(f"Training-only POD (r={pod.rank})")
    radii = np.asarray(manifest.get("candidate_spectral_radius", []), dtype=float)
    grid = np.asarray(manifest.get("candidate_grid", []), dtype=float)
    axes[2].plot(grid, radii, "o-", color=COLORS["pce"], ms=3.2, lw=1.0)
    axes[2].axhline(1.0, color="#888888", ls=":", lw=0.8)
    axes[2].set_xlabel(r"candidate $U_r$")
    axes[2].set_ylabel("spectral radius")
    axes[2].set_title("DMDc stability gate")
    fig.suptitle("VIV-PIV candidate library and external-test design", y=1.03, fontsize=9.2)
    save_pub(fig, out / "fig01_split_pod_candidates")


def figure_sensor_layout(config: dict[str, Any], data_root: pathlib.Path, out: pathlib.Path) -> None:
    path = list_cases(data_root)[str(config["test_cases"][0])]
    case = VIVCase.open(path)
    mask = np.asarray(case.mask[0] > 0.5)
    diameter_mm = float(config["cylinder_diameter_m"]) * 1000
    x = case.x_mm / diameter_mm
    y = case.y_mm / diameter_mm
    sensor_x = np.asarray(config["sensor_x_over_d"], dtype=float)
    sensor_y = np.asarray(config["sensor_y_over_d"], dtype=float)
    xx, yy = np.meshgrid(sensor_x, sensor_y)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.imshow(mask, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="Greys", alpha=0.19, aspect="auto")
    ax.scatter(xx.ravel(), yy.ravel(), s=8, facecolor="white", edgecolor=COLORS["pce"], lw=0.45, label="800 spatial PIV points")
    ax.scatter(0, 0, s=110, facecolor="#444444", edgecolor="white", lw=0.8, marker="o", label="cylinder reference")
    ax.set_xlim(-2.0, 8.9)
    ax.set_ylim(-2.9, 2.5)
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    ax.set_title("Sparse observation layout and valid wake region")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6)
    save_pub(fig, out / "fig02_sensor_layout")


def figure_metrics(summary_rows: list[dict[str, str]], out: pathlib.Path) -> None:
    rows = [
        r for r in summary_rows
        if r.get("method") in {"pce", "apce"}
        and r.get("status") == "completed"
        and int(r.get("sensor_density_points") or 800) == 800
        and str(r.get("uses_known_cylinder_displacement_input", "True")).lower() == "true"
    ]
    if not rows:
        return
    cases = sorted({r["case_id"] for r in rows})
    metrics = [("evaluation_nrmse", "held-out nRMSE"), ("normalized_crps", "normalized CRPS"), ("blackout_mean_nrmse", "blackout nRMSE")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True, gridspec_kw={"wspace": 0.33})
    x = np.arange(len(cases))
    width = 0.34
    for ax, (field, label) in zip(axes, metrics):
        for offset, method in [(-width / 2, "pce"), (width / 2, "apce")]:
            values = []
            for case in cases:
                candidates = [r for r in rows if r["case_id"] == case and r["method"] == method]
                values.append(float(candidates[0][field]) if candidates and candidates[0].get(field) not in {"", "None", "nan"} else np.nan)
            ax.bar(x + offset, values, width, color=COLORS[method], label=method.upper())
        ax.set_title(label)
        ax.set_xticks(x, cases, rotation=0)
        ax.grid(axis="y", color="#dddddd", lw=0.55)
    axes[0].set_ylabel("score (lower is better)")
    axes[0].legend(ncol=2, fontsize=6, loc="upper left")
    fig.suptitle("PCE/APCE held-out reconstruction and blackout admission", y=1.02, fontsize=9.2)
    save_pub(fig, out / "fig03_metrics")


def figure_calibration(summary_rows: list[dict[str, str]], out: pathlib.Path) -> None:
    rows = [
        r for r in summary_rows
        if r.get("method") in {"pce", "apce"}
        and r.get("status") == "completed"
        and int(r.get("sensor_density_points") or 800) == 800
        and str(r.get("uses_known_cylinder_displacement_input", "True")).lower() == "true"
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for method in ("pce", "apce"):
        selected = [r for r in rows if r["method"] == method]
        width = np.asarray([float(r["normalized_interval_width_90"]) for r in selected])
        coverage = np.asarray([float(r["coverage_90"]) for r in selected])
        ax.scatter(width, coverage, s=32, color=COLORS[method], label=method.upper(), zorder=3)
        annotation_offsets = {
            ("pce", "0463"): (4, 2), ("apce", "0463"): (4, 2),
            ("pce", "0556"): (4, 2), ("apce", "0556"): (4, 2),
            ("pce", "0679"): (4, -10), ("apce", "0679"): (4, 2),
            ("pce", "0803"): (4, 2), ("apce", "0803"): (4, 2),
            ("pce", "1359"): (4, 10), ("apce", "1359"): (4, -11),
        }
        for row, x_value, y_value in zip(selected, width, coverage):
            ax.annotate(row["case_id"], (x_value, y_value), xytext=annotation_offsets[(method, row["case_id"])], textcoords="offset points", fontsize=5.8)
    ax.axhline(0.90, color="#777777", lw=0.9, ls=":", label="nominal 90%")
    ax.set_xlabel("normalized 90% interval width")
    ax.set_ylabel("empirical 90% coverage")
    coverage_values = [float(r["coverage_90"]) for r in rows if r.get("coverage_90") not in {"", "None", "nan"}]
    ax.set_ylim(max(0.0, min(0.85, min(coverage_values) - 0.03)), min(1.02, max(0.97, max(coverage_values) + 0.02)))
    ax.set_title("Calibration-width trade-off")
    ax.grid(color="#eeeeee", lw=0.55)
    ax.legend(fontsize=6, loc="lower right")
    save_pub(fig, out / "fig03b_calibration_frontier")


def figure_weights_diagnostics(run_root: pathlib.Path, out: pathlib.Path, case_id: str = "0679") -> None:
    traces: dict[str, dict[str, np.ndarray]] = {}
    for method in ("pce", "apce"):
        path = preferred_trace(run_root, case_id, method)
        if path is not None and path.exists():
            traces[method] = load_trace(path)
    if not traces:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), gridspec_kw={"wspace": 0.28, "hspace": 0.34})
    for method, trace in traces.items():
        t = trace["time_s"][1:]
        weights = trace["weights"]
        grid = trace["candidate_grid"]
        axes[0, 0].plot(t, weights.max(axis=1), lw=1.2, color=COLORS[method], label=method.upper())
        axes[0, 1].plot(t, trace["separation"], lw=1.1, color=COLORS[method], label=method.upper())
        axes[1, 0].plot(t, trace["entropy"], lw=1.1, color=COLORS[method], label=method.upper())
        axes[1, 1].plot(t, 1.0 / np.sum(weights**2, axis=1), lw=1.1, color=COLORS[method], label=method.upper())
    axes[0, 0].set_title("maximum candidate weight")
    axes[0, 1].set_title("shadow score separation / sampling error")
    axes[1, 0].set_title("weight entropy")
    axes[1, 1].set_title("effective candidate count")
    for ax in axes.flat:
        ax.set_xlabel("time (s)")
        ax.grid(color="#eeeeee", lw=0.55)
    axes[0, 0].set_ylabel("weight")
    axes[0, 1].set_ylabel("ratio")
    axes[1, 0].set_ylabel("nats")
    axes[1, 1].set_ylabel("count")
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.suptitle(f"Shadow-anchored evidence diagnostics ({case_id}; operational scores)", y=0.99, fontsize=9.2)
    save_pub(fig, out / f"fig04_weights_diagnostics_{case_id}")


def figure_weight_maps(run_root: pathlib.Path, out: pathlib.Path, case_id: str = "0679") -> None:
    traces: dict[str, dict[str, np.ndarray]] = {}
    for method in ("pce", "apce"):
        path = preferred_trace(run_root, case_id, method)
        if path is not None and path.exists():
            traces[method] = load_trace(path)
    if set(traces) != {"pce", "apce"}:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), gridspec_kw={"wspace": 0.31, "hspace": 0.34})
    for ax, method in zip(axes[0], ("pce", "apce")):
        trace = traces[method]
        t = trace["time_s"][1:]
        grid = trace["candidate_grid"]
        weights = trace["weights"]
        image = ax.pcolormesh(t, grid, weights.T, shading="auto", cmap="viridis", vmin=0, vmax=1)
        ax.axhline(int(case_id) / 100, color="white", ls="--", lw=0.9, alpha=0.85)
        ax.set_title(f"{method.upper()} operational weight")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"candidate $U_r$")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="weight")
    for method, trace in traces.items():
        t = trace["time_s"][1:]
        weights = trace["weights"]
        grid = trace["candidate_grid"]
        axes[1, 0].plot(t, weights @ grid, lw=1.1, color=COLORS[method], label=method.upper())
        axes[1, 1].plot(t, trace["separation"], lw=1.05, color=COLORS[method], label=method.upper(), alpha=0.92)
    axes[1, 0].axhline(int(case_id) / 100, color="#777777", ls="--", lw=0.9, label="external case coordinate")
    axes[1, 0].set_title("evidence-weighted candidate coordinate")
    axes[1, 1].axhline(1.0, color="#777777", ls=":", lw=0.9, label="separation gate")
    axes[1, 1].set_title("shadow separation / score uncertainty")
    axes[1, 0].set_ylabel(r"weighted $U_r$")
    axes[1, 1].set_ylabel("ratio")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
        ax.grid(color="#eeeeee", lw=0.55)
        ax.legend(fontsize=5.8)
    fig.suptitle(f"Candidate evidence evolution for transition case {case_id}", y=0.99, fontsize=9.2)
    save_pub(fig, out / f"fig04b_weight_maps_{case_id}")


def figure_reconstruction(config: dict[str, Any], model_root: pathlib.Path, data_root: pathlib.Path, run_root: pathlib.Path, out: pathlib.Path, case_id: str = "0679") -> None:
    pod = PODModel.load(model_root / "pod_model.npz")
    case = VIVCase.open(list_cases(data_root)[case_id])
    frame = 700
    truth, valid = case.physical_frames(frame, frame + 1)
    truth = truth[0]
    valid = valid[0]
    rows: dict[str, np.ndarray] = {}
    for method in ("pce", "apce"):
        path = preferred_trace(run_root, case_id, method)
        if path is None or not path.exists():
            continue
        trace = load_trace(path)
        latent = trace["latent_estimate"][frame]
        decoded = pod.mean + latent @ pod.basis.T
        rows[method] = decoded.reshape(truth.shape)
    if not rows:
        return
    speed_truth = np.linalg.norm(truth, axis=-1)
    extent = [case.x_mm[0] / 50.0, case.x_mm[-1] / 50.0, case.y_mm[0] / 50.0, case.y_mm[-1] / 50.0]
    panels: list[tuple[str, np.ndarray, str]] = [("truth", speed_truth, "truth |v|")]
    for method in ("pce", "apce"):
        if method in rows:
            panels.append((method, np.linalg.norm(rows[method], axis=-1), f"{method.upper()} |v|"))
    if "apce" in rows:
        panels.append(("error", np.abs(np.linalg.norm(rows["apce"], axis=-1) - speed_truth), "APCE absolute error"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    speed_images = [image for name, image, _title in panels if name != "error"]
    speed_low = float(min(np.nanpercentile(image[valid], 1) for image in speed_images))
    speed_high = float(max(np.nanpercentile(image[valid], 99) for image in speed_images))
    for ax, (name, image, title) in zip(axes, panels):
        shown = np.ma.masked_where(~valid, image)
        cmap = "magma" if name != "error" else "viridis"
        limits = {} if name == "error" else {"vmin": speed_low, "vmax": speed_high}
        im = ax.imshow(shown, origin="lower", extent=extent, aspect="auto", cmap=cmap, **limits)
        ax.set_title(title)
        ax.set_xlabel(r"$x/D$")
        if ax is axes[0]:
            ax.set_ylabel(r"$y/D$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Held-out wake reconstruction at {case.time_s[frame]:.1f} s ({case_id})", y=1.02, fontsize=9.2)
    save_pub(fig, out / f"fig05_reconstruction_{case_id}")


def figure_energy_blackout(run_root: pathlib.Path, out: pathlib.Path, case_id: str = "0679") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"wspace": 0.32})
    found = False
    for method in ("pce", "apce"):
        path = preferred_trace(run_root, case_id, method)
        if path is None or not path.exists():
            continue
        trace = load_trace(path)
        axes[0].plot(trace["time_s"], trace["truth_energy"], color=COLORS["truth"], lw=1.0, alpha=0.75, label="truth" if method == "pce" else None)
        axes[0].plot(trace["time_s"], trace["predicted_energy"], color=COLORS[method], lw=1.1, label=method.upper())
        raw = json.loads(str(trace["blackout_rows_json"].item())) if "blackout_rows_json" in trace else []
        horizons = sorted({float(row["horizon_s"]) for row in raw})
        means = [np.mean([float(row["evaluation_nrmse"]) for row in raw if float(row["horizon_s"]) == h]) for h in horizons]
        stds = [np.std([float(row["evaluation_nrmse"]) for row in raw if float(row["horizon_s"]) == h]) for h in horizons]
        axes[1].errorbar(horizons, means, yerr=stds, marker="o", ms=3.5, lw=1.0, color=COLORS[method], capsize=2, label=method.upper())
        found = True
    if not found:
        plt.close(fig)
        return
    axes[0].set_title("Held-out kinetic-energy reconstruction")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("kinetic energy (data units)")
    axes[0].legend(ncol=3, fontsize=6)
    axes[1].set_title("Known-input blackout degradation")
    axes[1].set_xlabel("forecast horizon (s)")
    axes[1].set_ylabel("held-out nRMSE")
    axes[1].grid(color="#eeeeee", lw=0.55)
    axes[1].legend(fontsize=6)
    fig.suptitle(f"Energy and conditional short-time forecast ({case_id})", y=1.02, fontsize=9.2)
    save_pub(fig, out / f"fig06_energy_blackout_{case_id}")


def make_figures(config_path: pathlib.Path, variant: str | None = None) -> pathlib.Path:
    config = load_config(config_path)
    output_root = pathlib.Path(config["output_root"])
    variant = variant or f"rank{int(config['rank'])}_stride1"
    model_root = output_root / "models" / variant
    run_root = output_root / "runs" / variant
    summary_dir = output_root / "summaries" / variant
    out = output_root / "figures" / variant
    summary_rows = read_rows(summary_dir / "summary_metrics.csv")
    figure_split_pod_candidates(config, model_root, out)
    figure_sensor_layout(config, pathlib.Path(config["data_root"]), out)
    figure_metrics(summary_rows, out)
    figure_calibration(summary_rows, out)
    figure_weights_diagnostics(run_root, out)
    figure_weight_maps(run_root, out)
    figure_reconstruction(config, model_root, pathlib.Path(config["data_root"]), run_root, out)
    figure_energy_blackout(run_root, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Python Nature-style VIV-PIV figures.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    print(make_figures(args.config, args.variant))


if __name__ == "__main__":
    main()
