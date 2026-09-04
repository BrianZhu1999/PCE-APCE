"""Make full-duration 2x3 VIV-PIV field GIFs for the five held-out cases."""
from __future__ import annotations

import argparse
import pathlib

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


CASES = ("0463", "0556", "0679", "0803", "1359")
UR = {"0463": 4.63, "0556": 5.56, "0679": 6.79, "0803": 8.03, "1359": 13.59}


def trace_path(root: pathlib.Path, case_id: str, method: str, seed: int, layout: str) -> pathlib.Path:
    name = f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}_ens064_covfull_shr050.npz"
    return root / name


def extract_fields(
    case: VIVCase,
    pod: PODModel,
    source: str,
    latent: np.ndarray | None,
    component: str,
    device: torch.device,
    frame_stride: int,
    block: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    times: list[float] = []
    displacement: list[float] = []
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    selected = {"u": 0, "v": 1, "speed": 2}[component]
    for start, values, valid in case.iter_physical(block=block):
        stop = start + values.shape[0]
        if source == "truth":
            physical = values.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)
        else:
            state = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
            physical = (mean[None, :] + state @ basis.mT).detach().cpu().numpy()
            physical = physical.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)
        valid_grid = valid.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)[..., 0]
        if selected < 2:
            field = physical[..., selected]
        else:
            field = np.linalg.norm(physical, axis=-1)
        field = np.where(valid_grid, field, np.nan).astype(np.float32)
        keep = np.arange(values.shape[0]) % frame_stride == 0
        frames.extend(field[keep])
        times.extend(case.time_s[start:stop][keep].tolist())
        displacement.extend((case.cyl_displ_m[start:stop][keep] / 0.05).tolist())
    return np.asarray(frames), np.asarray(times), np.asarray(displacement)


def case_norm(field: np.ndarray, component: str) -> Normalize:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if component == "speed":
        return Normalize(vmin=0.0, vmax=max(float(np.max(finite)), 1e-8))
    limit = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-8)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def make_gif(
    fields: dict[str, np.ndarray],
    times: dict[str, np.ndarray],
    displacements: dict[str, np.ndarray],
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
    norms: dict[str, Normalize],
    component: str,
    source: str,
    output: pathlib.Path,
    frame_stride: int,
) -> None:
    cmap = "viridis" if component == "speed" else "RdBu_r"
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.25), constrained_layout=False)
    axes_flat = list(axes.flat)
    meshes = {}
    circles = {}
    for index, case_id in enumerate(CASES):
        ax = axes_flat[index]
        x, y = coordinates[case_id]
        first = fields[case_id][0]
        meshes[case_id] = ax.pcolormesh(x, y, first, shading="auto", cmap=cmap, norm=norms[case_id], rasterized=True)
        circles[case_id] = Circle((0.0, displacements[case_id][0]), 0.5,
                                  facecolor="white", edgecolor="#222222", linewidth=0.55, zorder=4)
        ax.add_patch(circles[case_id])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        ax.set_title(rf"$U_r={UR[case_id]:.2f}$", fontsize=8, pad=3)
        ax.tick_params(length=2, width=0.5, labelsize=5.6)
        if index // 3 == 1:
            ax.set_xlabel(r"$x/D$")
        else:
            ax.tick_params(labelbottom=False)
        if index % 3 == 0:
            ax.set_ylabel(r"$y/D$")
        else:
            ax.tick_params(labelleft=False)

    info = axes_flat[5]
    info.axis("off")
    info.text(0.5, 0.62, source.upper(), ha="center", va="center", fontsize=11, fontweight="bold")
    info.text(0.5, 0.42, component if component != "speed" else r"$|\mathbf{v}|$",
              ha="center", va="center", fontsize=10)
    info.text(0.5, 0.25, f"full duration\nframe stride = {frame_stride}",
              ha="center", va="center", fontsize=7.5, color="#444444")
    for case_id in CASES:
        cb = fig.colorbar(meshes[case_id], ax=axes_flat[CASES.index(case_id)],
                          fraction=0.046, pad=0.018)
        cb.ax.tick_params(labelsize=5.2, length=1.8, pad=1.0)
        cb.set_label("m s$^{-1}$", fontsize=5.4, labelpad=1.0)
    time_text = fig.text(0.5, 0.018, "", ha="center", va="bottom", fontsize=7.2)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.075, wspace=0.22, hspace=0.22)

    n_frames = min(fields[case_id].shape[0] for case_id in CASES)

    def update(frame_index: int):
        for case_id in CASES:
            meshes[case_id].set_array(fields[case_id][frame_index].ravel())
            circles[case_id].center = (0.0, float(displacements[case_id][frame_index]))
        time_text.set_text(f"t = {times[CASES[0]][frame_index]:.1f} s")
        return [*meshes.values(), *circles.values(), time_text]

    animation_obj = animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation_obj.save(output, writer=animation.PillowWriter(fps=10))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 2x3 full-time VIV-PIV GIFs.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--method", choices=("pce", "apce"), default="apce")
    parser.add_argument("--layout", default="adaptive_fullfield_valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--component", choices=("u", "v", "speed"), required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--source", choices=("truth", "reconstruction"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.frame_stride < 1:
        raise ValueError("frame stride must be positive")
    config = load_config(args.config)
    data_root = pathlib.Path(config["data_root"])
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    trace_root = result_root / "runs" / args.variant / "traces"
    cases = list_cases(data_root)
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    fields: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    displacements: dict[str, np.ndarray] = {}
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    norms: dict[str, Normalize] = {}
    for case_id in CASES:
        case = VIVCase.open(cases[case_id])
        latent = None
        if args.source == "reconstruction":
            with np.load(trace_path(trace_root, case_id, args.method, args.seed, args.layout), allow_pickle=False) as trace:
                latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
        field, time_s, displacement = extract_fields(
            case, pod, "truth" if args.source == "truth" else "prediction", latent,
            args.component, device, args.frame_stride,
        )
        fields[case_id] = field
        times[case_id] = time_s
        displacements[case_id] = displacement
        coordinates[case_id] = (np.asarray(case.x_mm) / 50.0, np.asarray(case.y_mm) / 50.0)
        norms[case_id] = case_norm(field, args.component)
    stem = args.output / f"viv_piv_fulltime_{args.source}_{args.method}_{args.component}_2x3.gif"
    make_gif(fields, times, displacements, coordinates, norms, args.component,
             args.source, stem, args.frame_stride)
    print(stem)


if __name__ == "__main__":
    main()
