"""Plot aspect-correct u/v wake fields for one VIV-PIV test condition."""
from __future__ import annotations

import argparse
import pathlib

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


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
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def trace_latent(run_root: pathlib.Path, case_id: str, method: str) -> np.ndarray:
    path = run_root / "traces" / f"viv_{case_id}_{method}_seed000_layout20x40_covfull.npz"
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["latent_estimate"], dtype=np.float32)


def frame_field(case: VIVCase, pod: PODModel, latent: np.ndarray, frame: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    values, valid = case.physical_frames(frame, frame + 1)
    truth = values[0]
    valid = valid[0]
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    state = torch.as_tensor(latent[frame], dtype=torch.float32, device=device)
    prediction = (mean + state @ basis.mT).cpu().numpy().reshape(truth.shape)
    return np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64), np.asarray(valid, dtype=bool)


def component_norm(values: list[np.ndarray], valid: np.ndarray) -> TwoSlopeNorm:
    merged = np.concatenate([v[valid] for v in values])
    scale = max(abs(float(np.nanpercentile(merged, 1))), abs(float(np.nanpercentile(merged, 99))), 1e-8)
    return TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)


def plot_panel(axis, field: np.ndarray, x: np.ndarray, y: np.ndarray, valid: np.ndarray, norm, cmap: str, title: str, circle_y: float) -> None:
    masked = np.where(valid, field, np.nan)
    mesh = axis.pcolormesh(x, y, masked, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    axis.add_patch(Circle((0.0, circle_y), 0.5, fill=False, ec="#303030", lw=0.55, zorder=4))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(float(y.min()), float(y.max()))
    axis.set_title(title, pad=3.0)
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$y/D$")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot u/v components for VIV-PIV case 0556.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--case", default="0556")
    parser.add_argument("--time", type=float, default=76.2)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    case_id = str(args.case).replace(",", "").zfill(4)[-4:]
    variant = args.variant or f"rank{int(config['rank'])}_stride1"
    model_root = pathlib.Path(config["output_root"]) / "models" / variant
    run_root = pathlib.Path(config["output_root"]) / "runs" / variant
    case = VIVCase.open(list_cases(pathlib.Path(config["data_root"]))[case_id])
    frame = int(np.argmin(np.abs(case.time_s - float(args.time))))
    actual_time = float(case.time_s[frame])
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    truth, pce, valid = frame_field(case, pod, trace_latent(run_root, case_id, "pce"), frame, device)
    _truth, apce, valid_apce = frame_field(case, pod, trace_latent(run_root, case_id, "apce"), frame, device)
    valid = valid & valid_apce
    diameter_mm = float(config["cylinder_diameter_m"]) * 1000.0
    x = case.x_mm / diameter_mm
    y = case.y_mm / diameter_mm
    circle_y = float(case.cyl_displ_m[frame] / float(config["cylinder_diameter_m"]))
    error = np.abs(apce - truth)

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), gridspec_kw={"wspace": 0.32, "hspace": 0.40})
    methods = [(truth, "truth"), (pce, "PCE"), (apce, "APCE"), (error, "APCE absolute error")]
    for row, component in enumerate((0, 1)):
        component_label = "u" if component == 0 else "v"
        component_norm_value = component_norm([truth[..., component], pce[..., component], apce[..., component]], valid)
        error_values = np.concatenate([error[..., component][valid]])
        error_norm = Normalize(vmin=0.0, vmax=max(float(np.nanpercentile(error_values, 99)), 1e-8))
        for column, (field, method_label) in enumerate(methods):
            current = field[..., component]
            norm = error_norm if method_label.endswith("error") else component_norm_value
            cmap = "magma" if method_label.endswith("error") else "RdBu_r"
            mesh = plot_panel(
                axes[row, column], current, x, y, valid, norm, cmap,
                f"{method_label} ({component_label})", circle_y,
            )
            if column == 0:
                axes[row, column].text(
                    -0.22, 0.5, component_label,
                    transform=axes[row, column].transAxes,
                    rotation=90, va="center", ha="center", fontsize=10, fontweight="bold",
                )
            colorbar = fig.colorbar(mesh, ax=axes[row, column], fraction=0.046, pad=0.025)
            colorbar.ax.tick_params(labelsize=6.5, length=2)
            colorbar.set_label("m s$^{-1}$" if not method_label.endswith("error") else "absolute error", fontsize=6.5)
    fig.suptitle(
        rf"Held-out velocity-component reconstruction at $t={actual_time:.1f}$ s ($U_r={int(case_id)/100:.2f}$)",
        y=0.995, fontsize=11, fontweight="bold",
    )
    fig.text(0.5, 0.012, "All panels use the measured x/D and y/D coordinates; aspect ratio is 1:1 in data units.", ha="center", fontsize=7)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"velocity_components_{case_id}_t{actual_time:.1f}_fullR"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    np.savez_compressed(
        args.output / f"velocity_components_{case_id}_t{actual_time:.1f}_fullR_source.npz",
        truth=truth, pce=pce, apce=apce, absolute_error=error, valid=valid,
        x_over_d=x, y_over_d=y, time_s=np.asarray(actual_time), frame=np.asarray(frame),
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
