"""Prepare and render four-method VIV-PIV blackout comparison GIFs."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import animation
from matplotlib.patches import Circle

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .assimilation import run_pass, run_two_pass
from .common import load_config, write_json
from .io import VIVCase, list_cases
from .rom import PODModel
from .run_case import _load_library, _load_scenario, blackout_origins


METHODS = ("pce", "apce", "aug_enkf", "bma")
METHOD_LABELS = {
    "pce": "PCE",
    "apce": "APCE",
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
}
METHOD_COLORS = {
    "pce": "#4C78A8",
    "apce": "#E45756",
    "aug_enkf": "#54A24B",
    "bma": "#B279A2",
}


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({name}) but is unavailable")
    return torch.device(name)


def _trace_path(root: pathlib.Path, case_id: str, method: str, seed: int, layout: str) -> pathlib.Path:
    return root / (
        f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}_ens064_covfull_shr050.npz"
    )


def _select_origin(
    config: dict[str, Any],
    scenario,
    trace_root: pathlib.Path,
    case_id: str,
    seed: int,
    layout: str,
) -> tuple[int, np.ndarray]:
    origins = np.asarray(sorted(blackout_origins(scenario, config)), dtype=np.int64)
    if origins.size == 0:
        raise RuntimeError(f"No admissible blackout origins for case {case_id}")
    reference_trace = _trace_path(trace_root, case_id, "apce", seed, layout)
    with np.load(reference_trace, allow_pickle=False) as trace:
        truth_energy = np.asarray(trace["truth_energy"], dtype=np.float64)
    origin = int(origins[int(np.argmax(truth_energy[origins]))])
    return origin, truth_energy


def _forecast_latent(
    result,
    origin: int,
    scenario,
    library,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> np.ndarray:
    snapshot = result.blackout_states[origin]
    dtype = torch.float64
    branches = torch.as_tensor(snapshot["branches"], dtype=dtype, device=device)
    weights = torch.as_tensor(snapshot["weights"], dtype=dtype, device=device)
    grid = np.asarray(snapshot["grid"], dtype=np.float64)
    member_coordinates = snapshot.get("member_coordinates")
    if member_coordinates is not None:
        coordinate_tensor = torch.as_tensor(member_coordinates, dtype=dtype, device=device)
        matrices, controls, q_sqrt = library.parameters_torch(coordinate_tensor, device, dtype)
    else:
        matrices, controls, q_sqrt = library.parameters(grid, device, dtype)
    q_sqrt = q_sqrt * float(scenario.process_noise_scale)
    inputs = torch.as_tensor(scenario.control, dtype=dtype, device=device)
    max_steps = int(round(max(config["blackout_horizons_s"]) / float(config["time_step_s"])))
    generator = torch.Generator(device=device).manual_seed(int(seed + 2_000_000 + origin))

    estimates: list[np.ndarray] = []
    initial = torch.sum(weights[:, None] * branches.mean(dim=1), dim=0)
    estimates.append(initial.detach().cpu().numpy())
    for step in range(1, max_steps + 1):
        noise = torch.randn(
            (branches.shape[1], library.rank), dtype=dtype, device=device, generator=generator
        )
        if member_coordinates is not None:
            members = branches[0]
            members = (
                torch.bmm(members.unsqueeze(1), matrices.transpose(-1, -2)).squeeze(1)
                + torch.einsum("nrc,c->nr", controls, inputs[origin + step - 1])
                + noise * q_sqrt
            )
            branches = members.unsqueeze(0)
        else:
            branches = (
                torch.matmul(branches, matrices.transpose(-1, -2))
                + torch.einsum("jrc,c->jr", controls, inputs[origin + step - 1]).unsqueeze(1)
                + noise.unsqueeze(0) * q_sqrt.unsqueeze(1)
            )
        estimate = torch.sum(weights[:, None] * branches.mean(dim=1), dim=0)
        estimates.append(estimate.detach().cpu().numpy())
    latent = np.asarray(estimates, dtype=np.float64)
    if not np.isfinite(latent).all():
        raise FloatingPointError("Blackout forecast emitted non-finite latent states")
    return latent


def prepare(args: argparse.Namespace) -> pathlib.Path:
    config = load_config(args.config)
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    trace_root = result_root / "runs" / args.variant / "traces"
    scenario = _load_scenario(
        model_root,
        args.case,
        config,
        False,
        None,
        args.layout,
        "full",
        covariance_shrinkage=float(config["observation_covariance_shrinkage"]),
    )
    origin, truth_energy = _select_origin(
        config, scenario, trace_root, args.case, args.seed, args.layout
    )
    library = _load_library(model_root)
    device = _device(args.device)
    if args.method in {"pce", "apce"}:
        result = run_two_pass(
            scenario,
            library,
            config,
            args.method,
            args.seed,
            device,
            record_trace=False,
            blackout_origins={origin},
        ).local
    else:
        result = run_pass(
            scenario,
            library,
            config,
            args.method,
            args.seed,
            device,
            record_trace=False,
            blackout_origins={origin},
        )
    latent = _forecast_latent(result, origin, scenario, library, config, args.seed, device)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"viv_{args.case}_{args.method}_seed{args.seed:03d}_blackout_source.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray(args.case),
        method=np.asarray(args.method),
        seed=np.asarray(args.seed),
        origin_index=np.asarray(origin),
        origin_time_s=np.asarray(scenario.time_s[origin]),
        horizon_s=np.arange(latent.shape[0], dtype=np.float64) * float(config["time_step_s"]),
        latent_estimate=latent,
        truth_energy_at_origin=np.asarray(truth_energy[origin]),
        candidate_grid=np.asarray(result.grid),
        frozen_weights=np.asarray(result.final_weights if not result.blackout_states else result.blackout_states[origin]["weights"]),
    )
    print(path)
    return path


def _decode_latent(pod: PODModel, latent: np.ndarray, device: torch.device) -> np.ndarray:
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    state = torch.as_tensor(latent, dtype=torch.float32, device=device)
    return (mean[None, :] + state @ basis.mT).detach().cpu().numpy()


def _nrmse_trace(
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    sensor_indices: np.ndarray,
) -> np.ndarray:
    truth_flat = truth.reshape(truth.shape[0], -1).astype(np.float64)
    pred_flat = prediction.reshape(prediction.shape[0], -1).astype(np.float64)
    valid_flat = np.repeat(valid[..., None], 2, axis=-1).reshape(valid.shape[0], -1)
    valid_flat[:, np.asarray(sensor_indices, dtype=np.int64)] = False
    diff = np.where(valid_flat, pred_flat - truth_flat, 0.0)
    reference = np.where(valid_flat, truth_flat, 0.0)
    denominator = np.sum(reference**2, axis=1)
    return np.sqrt(np.sum(diff**2, axis=1) / np.maximum(denominator, 1e-30))


def _speed(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(values, axis=-1)
    return np.where(valid, speed, np.nan).astype(np.float32)


def render(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    config = load_config(args.config)
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    case_path = list_cases(pathlib.Path(config["data_root"]))[args.case]
    case = VIVCase.open(case_path)
    pod = PODModel.load(model_root / "pod_model.npz")
    device = _device(args.device)

    sources: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        path = args.source / f"viv_{args.case}_{method}_seed{args.seed:03d}_blackout_source.npz"
        with np.load(path, allow_pickle=False) as archive:
            sources[method] = {key: np.asarray(archive[key]) for key in archive.files}
    origins = {int(source["origin_index"]) for source in sources.values()}
    if len(origins) != 1:
        raise RuntimeError(f"Methods use different blackout origins: {sorted(origins)}")
    origin = origins.pop()
    horizon = np.asarray(sources["apce"]["horizon_s"], dtype=np.float64)
    stop = origin + horizon.size
    truth, valid = case.physical_frames(origin, stop)
    ny, nx = truth.shape[1:3]
    predictions: dict[str, np.ndarray] = {}
    nrmse: dict[str, np.ndarray] = {}
    sensor_archive = np.load(
        model_root / "sensor_layouts" / args.layout / f"case_{args.case}.npz",
        allow_pickle=False,
    )
    sensor_indices = np.asarray(sensor_archive["sensor_flat_indices"], dtype=np.int64)
    for method in METHODS:
        flat = _decode_latent(pod, sources[method]["latent_estimate"], device)
        prediction = flat.reshape(horizon.size, ny, nx, 2)
        predictions[method] = prediction
        nrmse[method] = _nrmse_trace(truth, prediction, valid, sensor_indices)

    truth_speed = _speed(truth, valid)
    pred_speed = {method: _speed(values, valid) for method, values in predictions.items()}
    finite_parts = [truth_speed[np.isfinite(truth_speed)]]
    finite_parts.extend(values[np.isfinite(values)] for values in pred_speed.values())
    pooled = np.concatenate(finite_parts)
    vmax = max(float(np.percentile(pooled, 99.5)), 1e-8)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), constrained_layout=False)
    field_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    labels = ["Truth", "PCE", "APCE", "Aug-EnKF", "BMA"]
    fields = [truth_speed, pred_speed["pce"], pred_speed["apce"], pred_speed["aug_enkf"], pred_speed["bma"]]
    x_over_d = np.asarray(case.x_mm, dtype=float) / (float(config["cylinder_diameter_m"]) * 1000.0)
    y_over_d = np.asarray(case.y_mm, dtype=float) / (float(config["cylinder_diameter_m"]) * 1000.0)
    meshes = []
    circles = []
    for index, (ax, label, values) in enumerate(zip(field_axes, labels, fields)):
        mesh = ax.pcolormesh(
            x_over_d,
            y_over_d,
            values[0],
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        cylinder = Circle(
            (0.0, float(case.cyl_displ_m[origin]) / float(config["cylinder_diameter_m"])),
            0.5,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.65,
            zorder=5,
        )
        ax.add_patch(cylinder)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x_over_d.min()), float(x_over_d.max()))
        ax.set_ylim(float(y_over_d.min()), float(y_over_d.max()))
        ax.set_title(label, fontsize=8, pad=2.0)
        ax.tick_params(length=2.2, width=0.55, labelsize=6)
        if index in (3, 4):
            ax.set_xlabel(r"$x/D$")
        else:
            ax.tick_params(labelbottom=False)
        if index in (0, 3):
            ax.set_ylabel(r"$y/D$")
        else:
            ax.tick_params(labelleft=False)
        ax.text(-0.10, 1.03, chr(ord("a") + index), transform=ax.transAxes, fontsize=8, fontweight="bold")
        meshes.append(mesh)
        circles.append(cylinder)

    metric_ax = axes[1, 2]
    lines = {}
    for method in METHODS:
        (line,) = metric_ax.plot([], [], color=METHOD_COLORS[method], lw=1.25, label=METHOD_LABELS[method])
        lines[method] = line
    metric_ax.set_xlim(0.0, float(horizon[-1]))
    ymax = max(0.4, math.ceil(10.0 * max(float(np.max(values)) for values in nrmse.values())) / 10.0)
    metric_ax.set_ylim(0.0, ymax)
    metric_ax.set_xlabel("Blackout horizon (s)")
    metric_ax.set_ylabel("Unobserved full-field nRMSE")
    metric_ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.8)
    metric_ax.legend(loc="upper left", fontsize=6, ncol=2, handlelength=1.5, columnspacing=0.8)
    metric_ax.text(-0.10, 1.03, "f", transform=metric_ax.transAxes, fontsize=8, fontweight="bold")
    cursor = metric_ax.axvline(0.0, color="#555555", lw=0.8, ls="--")

    cax = fig.add_axes([0.075, 0.045, 0.545, 0.018])
    colorbar = fig.colorbar(meshes[0], cax=cax, orientation="horizontal")
    colorbar.set_label(r"Speed $|\mathbf{v}|$ (m s$^{-1}$)", fontsize=6.5, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=5.8, length=1.8, pad=1.2)
    status = fig.text(0.5, 0.976, "", ha="center", va="top", fontsize=8.5)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.925, bottom=0.115, wspace=0.20, hspace=0.12)

    def update(frame: int):
        current_displacement = float(case.cyl_displ_m[origin + frame]) / float(config["cylinder_diameter_m"])
        for mesh, circle, values in zip(meshes, circles, fields):
            mesh.set_array(values[frame].ravel())
            circle.center = (0.0, current_displacement)
        for method in METHODS:
            lines[method].set_data(horizon[: frame + 1], nrmse[method][: frame + 1])
        cursor.set_xdata([horizon[frame], horizon[frame]])
        status.set_text(
            rf"$U_r={int(args.case) / 100.0:.2f}$   blackout origin = {float(case.time_s[origin]):.2f} s"
            rf"   horizon = {horizon[frame]:.1f} s"
        )
        return [*meshes, *circles, *lines.values(), cursor, status]

    args.output.mkdir(parents=True, exist_ok=True)
    gif_path = args.output / f"viv_Ur{int(args.case) / 100.0:05.2f}_blackout_four_methods.gif"
    poster_path = args.output / f"viv_Ur{int(args.case) / 100.0:05.2f}_blackout_four_methods_poster.png"
    update(horizon.size - 1)
    fig.savefig(poster_path, dpi=240, bbox_inches="tight", facecolor="white")
    movie = animation.FuncAnimation(fig, update, frames=horizon.size, interval=125, blit=False)
    movie.save(gif_path, writer=animation.PillowWriter(fps=8))
    plt.close(fig)

    source_path = args.output / f"viv_Ur{int(args.case) / 100.0:05.2f}_blackout_source.npz"
    np.savez_compressed(
        source_path,
        horizon_s=horizon,
        truth_speed=truth_speed,
        pce_speed=pred_speed["pce"],
        apce_speed=pred_speed["apce"],
        aug_enkf_speed=pred_speed["aug_enkf"],
        bma_speed=pred_speed["bma"],
        pce_nrmse=nrmse["pce"],
        apce_nrmse=nrmse["apce"],
        aug_enkf_nrmse=nrmse["aug_enkf"],
        bma_nrmse=nrmse["bma"],
        x_over_d=x_over_d,
        y_over_d=y_over_d,
        cylinder_displacement_over_d=np.asarray(case.cyl_displ_m[origin:stop]) / float(config["cylinder_diameter_m"]),
    )
    metadata = {
        "case_id": args.case,
        "reduced_velocity": int(args.case) / 100.0,
        "seed": args.seed,
        "origin_selection": "maximum truth kinetic energy among the 20 predefined blackout origins",
        "origin_index": origin,
        "origin_time_s": float(case.time_s[origin]),
        "horizon_s": float(horizon[-1]),
        "frame_count": int(horizon.size),
        "fps": 8,
        "sensor_layout": args.layout,
        "effective_spatial_points": int(sensor_indices.size // 2),
        "metric": "u/v vector nRMSE on valid unobserved full-field scalar dimensions",
        "displayed_field": "physical speed magnitude",
        "color_limit_m_per_s": [0.0, vmax],
        "gif": str(gif_path),
        "poster": str(poster_path),
        "source_data": str(source_path),
    }
    write_json(args.output / f"viv_Ur{int(args.case) / 100.0:05.2f}_blackout_metadata.json", metadata)
    print(gif_path)
    return gif_path, poster_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--variant", default="rank256_stride1")
    prepare_parser.add_argument("--case", required=True)
    prepare_parser.add_argument("--method", choices=METHODS, required=True)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--layout", default="adaptive_fullfield_valid")
    prepare_parser.add_argument("--device", default="cuda:2")
    prepare_parser.add_argument("--output", type=pathlib.Path, required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--config", type=pathlib.Path, required=True)
    render_parser.add_argument("--variant", default="rank256_stride1")
    render_parser.add_argument("--case", required=True)
    render_parser.add_argument("--seed", type=int, default=0)
    render_parser.add_argument("--layout", default="adaptive_fullfield_valid")
    render_parser.add_argument("--device", default="cuda:2")
    render_parser.add_argument("--source", type=pathlib.Path, required=True)
    render_parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
