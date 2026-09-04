"""Plot the rank-256 PCE reconstruction for held-out VIV-PIV case 0463."""
from __future__ import annotations

import argparse
import pathlib

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 6.5,
    "axes.linewidth": 0.65,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


def symmetric_norm(*fields: np.ndarray, valid: np.ndarray) -> TwoSlopeNorm:
    values = np.concatenate([field[valid] for field in fields])
    scale = max(abs(float(np.nanpercentile(values, 1))), abs(float(np.nanpercentile(values, 99))), 1e-8)
    return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)


def add_field(
    axis: plt.Axes,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    field: np.ndarray,
    valid: np.ndarray,
    cylinder_y: float,
    norm: Normalize,
    cmap: str,
    title: str,
    colorbar_label: str,
) -> None:
    mesh = axis.pcolormesh(
        x_over_d,
        y_over_d,
        np.where(valid, field, np.nan),
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.add_patch(Circle((0.0, cylinder_y), 0.5, facecolor="white", edgecolor="#202020", linewidth=0.55, zorder=5))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(x_over_d.min()), float(x_over_d.max()))
    axis.set_ylim(float(y_over_d.min()), float(y_over_d.max()))
    axis.set_title(title, pad=2.0)
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$y/D$")
    divider = make_axes_locatable(axis)
    color_axis = divider.append_axes("right", size="3.2%", pad=0.035)
    colorbar = axis.figure.colorbar(mesh, cax=color_axis)
    colorbar.ax.tick_params(labelsize=5.4, length=1.8, width=0.5)
    colorbar.set_label(colorbar_label, fontsize=5.8, labelpad=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--case", default="0463")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    case_id = str(args.case).replace(",", "").zfill(4)[-4:]
    model_root = pathlib.Path(config["output_root"]) / "models" / args.variant
    run_root = pathlib.Path(config["output_root"]) / "runs" / args.variant
    trace_path = run_root / "traces" / (
        f"viv_{case_id}_pce_seed{args.seed:03d}_layout20x40_ens064_covfull_shr050.npz"
    )
    with np.load(trace_path, allow_pickle=False) as trace:
        time_s = np.asarray(trace["time_s"], dtype=np.float64)
        truth_energy = np.asarray(trace["truth_energy"], dtype=np.float64)
        predicted_energy = np.asarray(trace["predicted_energy"], dtype=np.float64)
        latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
    warmup_index = int(np.searchsorted(time_s, float(config["warmup_seconds"])))
    frame = warmup_index + int(np.nanargmax(truth_energy[warmup_index:]))

    case = VIVCase.open(list_cases(pathlib.Path(config["data_root"]))[case_id])
    values, valid_frames = case.physical_frames(frame, frame + 1)
    truth = np.asarray(values[0], dtype=np.float32)
    valid = np.asarray(valid_frames[0], dtype=bool)
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    state = torch.as_tensor(latent[frame], dtype=torch.float32, device=device)
    prediction = (mean + state @ basis.mT).cpu().numpy().reshape(truth.shape)
    absolute_error = np.abs(prediction - truth)

    diameter_mm = float(config["cylinder_diameter_m"]) * 1000.0
    x_over_d = np.asarray(case.x_mm, dtype=np.float64) / diameter_mm
    y_over_d = np.asarray(case.y_mm, dtype=np.float64) / diameter_mm
    cylinder_y = float(case.cyl_displ_m[frame] / float(config["cylinder_diameter_m"]))

    figure, axes = plt.subplots(2, 3, figsize=(7.18, 3.78))
    figure.subplots_adjust(left=0.062, right=0.985, bottom=0.115, top=0.935, wspace=0.24, hspace=0.10)
    panel_labels = iter("abcdef")
    for row, (component, symbol) in enumerate(((0, r"$u$"), (1, r"$v$"))):
        flow_norm = symmetric_norm(truth[..., component], prediction[..., component], valid=valid)
        error_scale = max(float(np.nanpercentile(absolute_error[..., component][valid], 99)), 1e-8)
        panels = (
            (truth[..., component], flow_norm, "RdBu_r", f"Measured {symbol}", r"m s$^{-1}$"),
            (prediction[..., component], flow_norm, "RdBu_r", f"PCE {symbol}", r"m s$^{-1}$"),
            (absolute_error[..., component], Normalize(0.0, error_scale), "magma", rf"$|{symbol}_{{PCE}}-{symbol}_{{obs}}|$", r"m s$^{-1}$"),
        )
        for column, panel in enumerate(panels):
            axis = axes[row, column]
            add_field(axis, x_over_d, y_over_d, panel[0], valid, cylinder_y, panel[1], panel[2], panel[3], panel[4])
            axis.text(-0.12, 1.02, next(panel_labels), transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")
            if column > 0:
                axis.set_ylabel("")
                axis.tick_params(labelleft=False)

    figure.text(
        0.5,
        0.025,
        rf"Held-out $U_r=4.63$; rank 256; $t={time_s[frame]:.3f}$ s (measured kinetic-energy maximum); sequence nRMSE = 0.1698",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#3F3F3F",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "case0463_rank256_pce_uv"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    np.savez_compressed(
        args.output / "case0463_rank256_pce_uv_source.npz",
        truth=truth,
        prediction=prediction,
        absolute_error=absolute_error,
        valid=valid,
        x_over_d=x_over_d,
        y_over_d=y_over_d,
        frame=np.asarray(frame),
        time_s=np.asarray(time_s[frame]),
        truth_energy=np.asarray(truth_energy[frame]),
        predicted_energy=np.asarray(predicted_energy[frame]),
        sequence_nrmse=np.asarray(0.1698416541760394),
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
