"""Create one full-duration 2 x 3 truth/APCE GIF for each held-out VIV case."""
from __future__ import annotations

import argparse
import json
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
COMPONENTS = ("u", "v", "speed")


def trace_path(
    root: pathlib.Path,
    case_id: str,
    method: str,
    seed: int,
    layout: str,
    *,
    include_ensemble_suffix: bool,
) -> pathlib.Path:
    ensemble_suffix = "_ens064" if include_ensemble_suffix else ""
    filename = f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}{ensemble_suffix}_covfull_shr050.npz"
    return root / filename


def extract_movie_fields(
    case: VIVCase,
    pod: PODModel,
    latent: np.ndarray,
    device: torch.device,
    frame_stride: int,
    block: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read each raw block once and decode the paired APCE block once."""
    truth_frames: list[np.ndarray] = []
    reconstruction_frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)

    for start, values, valid in case.iter_physical(block=block):
        stop = start + values.shape[0]
        local_indices = np.arange(values.shape[0])
        keep = local_indices[(start + local_indices) % frame_stride == 0]
        if keep.size == 0:
            continue

        truth = values.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)
        state = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
        reconstruction = (mean[None, :] + state @ basis.mT).detach().cpu().numpy()
        reconstruction = reconstruction.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)
        valid_grid = valid.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)[..., 0]

        truth = np.where(valid_grid[..., None], truth, np.nan).astype(np.float32)
        reconstruction = np.where(valid_grid[..., None], reconstruction, np.nan).astype(np.float32)
        truth_frames.append(truth[keep])
        reconstruction_frames.append(reconstruction[keep])
        times.append(np.asarray(case.time_s[start:stop][keep], dtype=np.float64))
        displacements.append(np.asarray(case.cyl_displ_m[start:stop][keep] / 0.05, dtype=np.float64))

    truth_uv = np.concatenate(truth_frames, axis=0)
    reconstruction_uv = np.concatenate(reconstruction_frames, axis=0)
    return (
        truth_uv,
        reconstruction_uv,
        np.concatenate(times),
        np.concatenate(displacements),
    )


def component_fields(uv: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "u": uv[..., 0],
        "v": uv[..., 1],
        "speed": np.linalg.norm(uv, axis=-1).astype(np.float32),
    }


def shared_norm(truth: np.ndarray, reconstruction: np.ndarray, component: str) -> Normalize:
    values = np.concatenate((truth[np.isfinite(truth)], reconstruction[np.isfinite(reconstruction)]))
    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    if component == "speed":
        return Normalize(vmin=0.0, vmax=max(float(np.max(values)), 1e-8))
    limit = max(abs(float(np.min(values))), abs(float(np.max(values))), 1e-8)
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def norm_limits(norm: Normalize) -> dict[str, float]:
    return {"vmin": float(norm.vmin), "vmax": float(norm.vmax)}


def make_case_gif(
    case_id: str,
    truth_fields: dict[str, np.ndarray],
    reconstruction_fields: dict[str, np.ndarray],
    times: np.ndarray,
    displacements: np.ndarray,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    output: pathlib.Path,
    frame_stride: int,
) -> dict[str, object]:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    norms = {
        component: shared_norm(truth_fields[component], reconstruction_fields[component], component)
        for component in COMPONENTS
    }
    cmaps = {"u": "RdBu_r", "v": "RdBu_r", "speed": "viridis"}
    titles = {"u": r"$u$", "v": r"$v$", "speed": r"$|\mathbf{v}|$"}

    fig = plt.figure(figsize=(11.4, 4.25), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=(1.0, 0.035, 1.0, 0.035, 1.0, 0.035),
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.89,
        wspace=0.12,
        hspace=0.12,
    )
    axes = np.empty((2, 3), dtype=object)
    color_axes = []
    meshes: dict[tuple[int, int], object] = {}
    circles: dict[tuple[int, int], Circle] = {}
    row_sources = (truth_fields, reconstruction_fields)
    for column, component in enumerate(COMPONENTS):
        color_axis = fig.add_subplot(grid[:, 2 * column + 1])
        color_axes.append(color_axis)
        for row in range(2):
            ax = fig.add_subplot(grid[row, 2 * column])
            axes[row, column] = ax
            mesh = ax.pcolormesh(
                x_over_d,
                y_over_d,
                row_sources[row][component][0],
                shading="auto",
                cmap=cmaps[component],
                norm=norms[component],
                rasterized=True,
            )
            meshes[(row, column)] = mesh
            circle = Circle(
                (0.0, float(displacements[0])),
                0.5,
                facecolor="white",
                edgecolor="#202020",
                linewidth=0.65,
                zorder=4,
            )
            circles[(row, column)] = circle
            ax.add_patch(circle)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(float(x_over_d.min()), float(x_over_d.max()))
            ax.set_ylim(float(y_over_d.min()), float(y_over_d.max()))
            ax.tick_params(length=2.0, width=0.5, labelsize=5.7, pad=1.5)
            if row == 0:
                ax.set_title(titles[component], fontsize=8.2, pad=3.0)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$x/D$", fontsize=6.7, labelpad=1.5)
            if column == 0:
                ax.set_ylabel(r"$y/D$", fontsize=6.7, labelpad=1.5)
            else:
                ax.tick_params(labelleft=False)
        colorbar = fig.colorbar(meshes[(0, column)], cax=color_axis)
        colorbar.set_label(r"m s$^{-1}$", fontsize=6.3, labelpad=2.5)
        colorbar.ax.tick_params(labelsize=5.5, length=2.0, width=0.5, pad=1.5)

    fig.text(0.015, 0.683, "Ground truth", rotation=90, ha="center", va="center", fontsize=7.3)
    fig.text(0.015, 0.300, "APCE reconstruction", rotation=90, ha="center", va="center", fontsize=7.3)
    case_text = fig.text(0.075, 0.955, rf"$U_r={UR[case_id]:.2f}$", ha="left", va="top", fontsize=9.0)
    time_text = fig.text(0.985, 0.955, "", ha="right", va="top", fontsize=8.0)

    def update(frame_index: int):
        for column, component in enumerate(COMPONENTS):
            for row in range(2):
                meshes[(row, column)].set_array(row_sources[row][component][frame_index].ravel())
                circles[(row, column)].center = (0.0, float(displacements[frame_index]))
        time_text.set_text(
            rf"$t={times[frame_index]:.1f}\ \mathrm{{s}},\quad y_c/D={displacements[frame_index]:+.3f}$"
        )
        return [*meshes.values(), *circles.values(), case_text, time_text]

    movie = animation.FuncAnimation(fig, update, frames=times.size, interval=100, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer=animation.PillowWriter(fps=10))
    plt.close(fig)
    return {
        "case_id": case_id,
        "reduced_velocity": UR[case_id],
        "frames": int(times.size),
        "frame_stride": int(frame_stride),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "color_limits_m_per_s": {component: norm_limits(norms[component]) for component in COMPONENTS},
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create five full-time 2 x 3 truth/APCE VIV-PIV GIFs.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--method", choices=("pce", "apce"), default="apce")
    parser.add_argument("--layout", default="adaptive_fullfield_valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument(
        "--omit-ensemble-suffix",
        action="store_true",
        help="Use run ids without the explicit _ens064 token.",
    )
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
    case_paths = list_cases(data_root)
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    selected_cases = tuple(args.case) if args.case else CASES

    metadata = []
    for case_id in selected_cases:
        case = VIVCase.open(case_paths[case_id])
        with np.load(
            trace_path(
                trace_root,
                case_id,
                args.method,
                args.seed,
                args.layout,
                include_ensemble_suffix=not args.omit_ensemble_suffix,
            ),
            allow_pickle=False,
        ) as trace:
            latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
        truth_uv, reconstruction_uv, times, displacements = extract_movie_fields(
            case, pod, latent, device, args.frame_stride
        )
        truth_fields = component_fields(truth_uv)
        reconstruction_fields = component_fields(reconstruction_uv)
        output = args.output / (
            f"viv_piv_Ur{UR[case_id]:05.2f}_truth_vs_{args.method}_"
            f"{args.layout}_fulltime_2x3.gif"
        )
        item = make_case_gif(
            case_id,
            truth_fields,
            reconstruction_fields,
            times,
            displacements,
            np.asarray(case.x_mm) / 50.0,
            np.asarray(case.y_mm) / 50.0,
            output,
            args.frame_stride,
        )
        metadata.append(item)
        print(output, flush=True)

    metadata_path = args.output / "viv_piv_fulltime_gif_metadata.json"
    existing = []
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    updated = {item["case_id"]: item for item in [*existing, *metadata]}
    metadata_path.write_text(
        json.dumps(list(updated.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metadata_path, flush=True)


if __name__ == "__main__":
    main()
