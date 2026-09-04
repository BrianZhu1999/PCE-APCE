"""Make a full-time five-case truth/APCE vorticity comparison GIF."""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


CASES = ("0463", "0556", "0679", "0803", "1359")
UR = {"0463": 4.63, "0556": 5.56, "0679": 6.79, "0803": 8.03, "1359": 13.59}


def trace_path(root: pathlib.Path, case_id: str, method: str, seed: int, layout: str) -> pathlib.Path:
    return root / f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}_ens064_covfull_shr050.npz"


def mask_aware_vorticity(
    uv: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    smoothing_sigma: float,
) -> np.ndarray:
    """Smooth mask-normalized velocity, then compute omega=dv/dx-du/dy."""
    sigma = (0.0, float(smoothing_sigma), float(smoothing_sigma))
    weight = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    smoothed = np.empty_like(uv, dtype=np.float32)
    for component in (0, 1):
        numerator = gaussian_filter(
            np.where(valid, uv[..., component], 0.0).astype(np.float32),
            sigma=sigma,
            mode="nearest",
        )
        smoothed[..., component] = numerator / np.maximum(weight, 1e-6)
    smooth_valid = valid & (weight > 0.5)
    safe = np.where(smooth_valid[..., None], smoothed, 0.0)
    u = safe[..., 0]
    v = safe[..., 1]
    du_dy = np.gradient(u, y_m, axis=1)
    dv_dx = np.gradient(v, x_m, axis=2)
    omega = dv_dx - du_dy
    finite = np.isfinite(smoothed).all(axis=-1)
    interior = smooth_valid & finite
    interior[:, :, 0] = False
    interior[:, :, -1] = False
    interior[:, 0, :] = False
    interior[:, -1, :] = False
    interior &= np.roll(smooth_valid, 1, axis=1) & np.roll(smooth_valid, -1, axis=1)
    interior &= np.roll(smooth_valid, 1, axis=2) & np.roll(smooth_valid, -1, axis=2)
    return np.where(interior, omega, np.nan).astype(np.float32)


def extract_case(
    case: VIVCase,
    pod: PODModel,
    latent: np.ndarray,
    device: torch.device,
    frame_stride: int,
    smoothing_sigma: float,
    block: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_m = np.asarray(case.x_mm, dtype=np.float64) / 1000.0
    y_m = np.asarray(case.y_mm, dtype=np.float64) / 1000.0
    truth_frames: list[np.ndarray] = []
    prediction_frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    displacement: list[np.ndarray] = []
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    for start, values, valid_flat in case.iter_physical(block=block):
        stop = start + values.shape[0]
        local = np.arange(values.shape[0])
        keep = local[(start + local) % frame_stride == 0]
        if keep.size == 0:
            continue
        truth_uv = values.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2).astype(np.float32)
        valid = valid_flat.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)[..., 0]
        prediction_flat = (
            torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device) @ basis.mT + mean[None, :]
        ).detach().cpu().numpy()
        prediction_uv = prediction_flat.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)
        truth_frames.append(mask_aware_vorticity(truth_uv, valid, x_m, y_m, smoothing_sigma)[keep])
        prediction_frames.append(mask_aware_vorticity(prediction_uv, valid, x_m, y_m, smoothing_sigma)[keep])
        times.append(np.asarray(case.time_s[start:stop][keep], dtype=np.float64))
        displacement.append(np.asarray(case.cyl_displ_m[start:stop][keep] / 0.05, dtype=np.float64))
    return (
        np.concatenate(truth_frames, axis=0),
        np.concatenate(prediction_frames, axis=0),
        np.concatenate(times),
        np.concatenate(displacement),
    )


def make_gif(
    fields: dict[str, tuple[np.ndarray, np.ndarray]],
    times: dict[str, np.ndarray],
    displacement: dict[str, np.ndarray],
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    output: pathlib.Path,
    frame_stride: int,
    smoothing_sigma: float,
    color_percentile: float,
) -> dict[str, object]:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(15.4, 4.25), constrained_layout=False)
    grid = fig.add_gridspec(
        2, 10,
        width_ratios=sum(([1.0, 0.035] for _ in CASES), []),
        left=0.048, right=0.993, bottom=0.115, top=0.89,
        wspace=0.10, hspace=0.12,
    )
    cmap = "RdBu_r"
    axes: dict[tuple[int, str], object] = {}
    meshes: dict[tuple[int, str], object] = {}
    circles: dict[tuple[int, str], Circle] = {}
    norms: dict[str, TwoSlopeNorm] = {}
    for col, case_id in enumerate(CASES):
        truth, prediction = fields[case_id]
        combined = np.concatenate((truth[np.isfinite(truth)], prediction[np.isfinite(prediction)]))
        limit = max(float(np.percentile(np.abs(combined), color_percentile)), 1e-8)
        norms[case_id] = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        for row, source in enumerate((truth, prediction)):
            ax = fig.add_subplot(grid[row, 2 * col])
            axes[(row, case_id)] = ax
            mesh = ax.imshow(
                source[0], origin="lower",
                extent=(float(x_over_d.min()), float(x_over_d.max()), float(y_over_d.min()), float(y_over_d.max())),
                cmap=cmap, norm=norms[case_id], interpolation="bilinear", rasterized=True,
            )
            meshes[(row, case_id)] = mesh
            circle = Circle((0.0, displacement[case_id][0]), 0.5,
                            facecolor="white", edgecolor="#202020", linewidth=0.6, zorder=4)
            circles[(row, case_id)] = circle
            ax.add_patch(circle)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(x_over_d.min()), float(x_over_d.max()))
            ax.set_ylim(float(y_over_d.min()), float(y_over_d.max()))
            ax.tick_params(length=2, width=0.5, labelsize=5.5, pad=1.2)
            if row == 0:
                ax.set_title(rf"$U_r={UR[case_id]:.2f}$", fontsize=8.0, pad=3.0)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$x/D$", fontsize=6.5, labelpad=1.3)
            if col == 0:
                ax.set_ylabel(r"$y/D$", fontsize=6.5, labelpad=1.2)
            else:
                ax.tick_params(labelleft=False)
        cax = fig.add_subplot(grid[:, 2 * col + 1])
        cb = fig.colorbar(meshes[(0, case_id)], cax=cax)
        cb.set_label(r"$\omega$ (s$^{-1}$)", fontsize=6.0, labelpad=2.0)
        cb.ax.tick_params(labelsize=5.0, length=1.8, width=0.45, pad=1.2)
    fig.text(0.012, 0.685, "Ground truth", rotation=90, ha="center", va="center", fontsize=7.0)
    fig.text(0.012, 0.300, "APCE reconstruction", rotation=90, ha="center", va="center", fontsize=7.0)
    time_text = fig.text(0.993, 0.953, "", ha="right", va="top", fontsize=7.8)
    n_frames = min(fields[case_id][0].shape[0] for case_id in CASES)
    t0 = {case_id: float(times[case_id][0]) for case_id in CASES}

    def update(frame: int):
        for case_id in CASES:
            truth, prediction = fields[case_id]
            meshes[(0, case_id)].set_data(truth[frame])
            meshes[(1, case_id)].set_data(prediction[frame])
            circles[(0, case_id)].center = (0.0, float(displacement[case_id][frame]))
            circles[(1, case_id)].center = (0.0, float(displacement[case_id][frame]))
        relative_times = [times[case_id][frame] - t0[case_id] for case_id in CASES]
        time_text.set_text(rf"$t-t_0={float(np.mean(relative_times)):.1f}\ \mathrm{{s}},\quad frame={frame + 1}/{n_frames}$")
        return [*meshes.values(), *circles.values(), time_text]

    movie = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    return {
        "cases": list(CASES), "frames": int(n_frames), "frame_stride": int(frame_stride),
        "vorticity_units": "s^-1",
        "smoothing": {"type": "mask-normalized Gaussian on u/v before differentiation", "sigma_grid_points": float(smoothing_sigma)},
        "color_scale": {"type": "fixed symmetric per case", "absolute_percentile": float(color_percentile)},
        "color_limits_s^-1": {
            case_id: {"vmin": float(norms[case_id].vmin), "vmax": float(norms[case_id].vmax)} for case_id in CASES
        },
        "output": output.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a five-case full-time vorticity GIF.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--method", choices=("pce", "apce"), default="apce")
    parser.add_argument("--layout", default="adaptive_fullfield_valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--color-percentile", type=float, default=99.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = pathlib.Path(config["data_root"])
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    trace_root = result_root / "runs" / args.variant / "traces"
    cases = list_cases(data_root)
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    times: dict[str, np.ndarray] = {}
    displacement: dict[str, np.ndarray] = {}
    x_over_d: np.ndarray | None = None
    y_over_d: np.ndarray | None = None
    for case_id in CASES:
        case = VIVCase.open(cases[case_id])
        with np.load(trace_path(trace_root, case_id, args.method, args.seed, args.layout), allow_pickle=False) as trace:
            latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
        truth, prediction, time_s, y_c = extract_case(
            case, pod, latent, device, args.frame_stride, args.smoothing_sigma
        )
        fields[case_id] = (truth, prediction)
        times[case_id] = time_s
        displacement[case_id] = y_c
        x_over_d = np.asarray(case.x_mm, dtype=np.float64) / 50.0
        y_over_d = np.asarray(case.y_mm, dtype=np.float64) / 50.0
    assert x_over_d is not None and y_over_d is not None
    output = args.output / f"viv_piv_vorticity_smoothed_truth_vs_{args.method}_5cases_fulltime.gif"
    metadata = make_gif(
        fields, times, displacement, x_over_d, y_over_d, output,
        args.frame_stride, args.smoothing_sigma, args.color_percentile,
    )
    metadata_path = args.output / "viv_piv_vorticity_smoothed_gif_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(metadata_path)


if __name__ == "__main__":
    main()
