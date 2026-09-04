"""Plot measured velocity components and reconstructed speed for five held-out VIV cases."""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


CASES = ("0463", "0556", "0679", "0803", "1359")
UR = {"0463": 4.63, "0556": 5.56, "0679": 6.79, "0803": 8.03, "1359": 13.59}


def _trace_path(root: pathlib.Path, case_id: str, method: str, seed: int, layout: str) -> pathlib.Path:
    return root / (
        f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}_ens064_covfull_shr050.npz"
    )


def _decode(pod: PODModel, latent: np.ndarray, shape: tuple[int, ...], device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
        mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
        state = torch.as_tensor(latent, dtype=torch.float32, device=device)
        return (mean + state @ basis.mT).cpu().numpy().reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--region", choices=("wake", "full_box", "full"), default="full",
                        help="wake-only crop, full field with wake box, or unboxed full field")
    parser.add_argument("--method", choices=("pce", "apce"), default="apce")
    parser.add_argument("--layout", default="adaptive_fullfield_valid")
    parser.add_argument("--name", default="fig07_five_cases_uv_synthetic_speed",
                        help="output filename stem inside --output")
    args = parser.parse_args()

    config = load_config(args.config)
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    trace_root = result_root / "runs" / args.variant / "traces"
    pod = PODModel.load(model_root / "pod_model.npz")
    cases = list_cases(pathlib.Path(config["data_root"]))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    diameter_m = float(config["cylinder_diameter_m"])
    warmup_s = float(config["warmup_seconds"])

    records = []
    component_values = []
    speed_values = []
    for case_id in CASES:
        case = VIVCase.open(cases[case_id])
        warmup = int(np.searchsorted(case.time_s, warmup_s))
        centred = case.cyl_displ_m - np.median(case.cyl_displ_m[warmup:])
        frame = warmup + int(np.argmax(np.abs(centred[warmup:])))
        truth_block, valid_block = case.physical_frames(frame, frame + 1)
        truth = np.asarray(truth_block[0], dtype=np.float32)
        valid = np.asarray(valid_block[0], dtype=bool)
        with np.load(_trace_path(trace_root, case_id, args.method, args.seed, args.layout), allow_pickle=False) as trace:
            latent = np.asarray(trace["latent_estimate"][frame], dtype=np.float32)
        prediction = _decode(pod, latent, truth.shape, device)
        truth_speed = np.linalg.norm(truth, axis=-1)
        prediction_speed = np.linalg.norm(prediction, axis=-1)
        component_values.extend((truth[..., 0][valid], truth[..., 1][valid],
                                 prediction[..., 0][valid], prediction[..., 1][valid]))
        speed_values.extend((truth_speed[valid], prediction_speed[valid]))
        records.append({
            "case_id": case_id,
            "frame": frame,
            "time_s": float(case.time_s[frame]),
            "truth": truth,
            "valid": valid,
            "prediction": prediction,
            "truth_speed": truth_speed,
            "prediction_speed": prediction_speed,
            "x": np.asarray(case.x_mm, dtype=float) / (diameter_m * 1000.0),
            "y": np.asarray(case.y_mm, dtype=float) / (diameter_m * 1000.0),
            "cylinder_y": float(case.cyl_displ_m[frame] / diameter_m),
        })

    components = np.concatenate(component_values)
    component_scale = max(abs(float(np.nanpercentile(components, 0.5))),
                          abs(float(np.nanpercentile(components, 99.5))), 1e-8)
    speed_scale = max(float(np.nanpercentile(np.concatenate(speed_values), 99.5)), 1e-8)
    component_norm = TwoSlopeNorm(vmin=-component_scale, vcenter=0.0, vmax=component_scale)
    speed_norm = Normalize(vmin=0.0, vmax=speed_scale)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 10.0), constrained_layout=False)
    grid = fig.add_gridspec(
        15, 3,
        height_ratios=[0.16, 1.0, 1.0] * 5,
        left=0.105, right=0.970, bottom=0.090, top=0.950,
        wspace=0.035, hspace=0.035,
    )
    axes = np.asarray([
        [fig.add_subplot(grid[3 * case_index + method_index + 1, col])
         for col in range(3)]
        for case_index in range(5)
        for method_index in range(2)
    ])
    for case_index, record in enumerate(records):
        label_axis = fig.add_subplot(grid[3 * case_index, :])
        label_axis.axis("off")
        label_axis.text(0.5, 0.48, rf"$U_r={UR[record['case_id']]:.2f}$",
                        ha="center", va="center", fontsize=7.2)
    headers = (r"$u$", r"$v$", r"$|\mathbf{v}|$")
    meshes = []
    for case_index, record in enumerate(records):
        row_fields = (
            (record["truth"][..., 0], record["truth"][..., 1], record["truth_speed"]),
            (record["prediction"][..., 0], record["prediction"][..., 1], record["prediction_speed"]),
        )
        for method_index, fields in enumerate(row_fields):
            row = case_index * 2 + method_index
            for col, (axis, field) in enumerate(zip(axes[row], fields)):
                is_speed = col == 2
                norm = speed_norm if is_speed else component_norm
                cmap = "viridis" if is_speed else "RdBu_r"
                mesh = axis.pcolormesh(record["x"], record["y"],
                                       np.where(record["valid"], field, np.nan),
                                       shading="auto", cmap=cmap, norm=norm, rasterized=True)
                meshes.append(mesh)
                axis.add_patch(Circle((0.0, record["cylinder_y"]), 0.5, facecolor="white",
                                      edgecolor="#222222", linewidth=0.45, zorder=4))
                axis.set_aspect("equal", adjustable="box")
                if args.region == "wake":
                    axis.set_xlim(0.85, 8.1)
                    axis.set_ylim(-2.1, 2.1)
                    axis.axvline(1.0, color="#263238", linewidth=0.45,
                                 linestyle=(0, (2, 2)), alpha=0.75, zorder=5)
                elif args.region == "full_box":
                    axis.set_xlim(record["x"].min(), record["x"].max())
                    axis.set_ylim(record["y"].min(), record["y"].max())
                    axis.add_patch(Rectangle((1.0, -2.0), 7.0, 4.0,
                                             facecolor="none", edgecolor="#263238",
                                             linewidth=0.55, linestyle=(0, (2, 2)),
                                             zorder=5))
                else:
                    axis.set_xlim(record["x"].min(), record["x"].max())
                    axis.set_ylim(record["y"].min(), record["y"].max())
                axis.tick_params(length=1.8, width=0.5)
                if row < 9:
                    axis.tick_params(labelbottom=False)
                else:
                    axis.set_xlabel(r"$x/D$")
                if col == 0:
                    method_label = "Measured" if method_index == 0 else args.method.upper()
                    axis.set_ylabel(
                        method_label + "\n" + r"$y/D$"
                    )
                else:
                    axis.tick_params(labelleft=False)

    fig.canvas.draw()
    for col, header in enumerate(headers):
        position = axes[0, col].get_position()
        fig.text(0.5 * (position.x0 + position.x1), 0.972, header,
                 ha="center", va="center", fontsize=7.0)

    cax_uv = fig.add_axes([0.105, 0.032, 0.565, 0.011])
    cb_uv = fig.colorbar(meshes[0], cax=cax_uv, orientation="horizontal")
    cb_uv.set_label(r"velocity (m s$^{-1}$)", fontsize=6.0, labelpad=1.5)
    cb_uv.ax.tick_params(labelsize=5.5, length=1.8, pad=1.0)
    cax_speed = fig.add_axes([0.690, 0.032, 0.280, 0.011])
    cb_speed = fig.colorbar(meshes[2], cax=cax_speed, orientation="horizontal")
    cb_speed.set_label(r"speed (m s$^{-1}$)", fontsize=6.0, labelpad=1.5)
    cb_speed.ax.tick_params(labelsize=5.5, length=1.8, pad=1.0)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / args.name
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    source = {"component_scale": np.asarray(component_scale), "speed_scale": np.asarray(speed_scale)}
    metadata = {"seed": args.seed, "variant": args.variant, "region": args.region,
                "method": args.method, "sensor_layout": args.layout,
                "wake_roi_over_d": [1.0, 8.0, -2.0, 2.0],
                "representative_frame_rule":
                "maximum absolute centred cylinder displacement after warm-up", "cases": []}
    for record in records:
        cid = record["case_id"]
        source[f"{cid}_truth"] = record["truth"]
        source[f"{cid}_valid"] = record["valid"]
        source[f"{cid}_{args.method}"] = record["prediction"]
        source[f"{cid}_truth_speed"] = record["truth_speed"]
        source[f"{cid}_{args.method}_speed"] = record["prediction_speed"]
        source[f"{cid}_x_over_d"] = record["x"]
        source[f"{cid}_y_over_d"] = record["y"]
        metadata["cases"].append({"case_id": cid, "reduced_velocity": UR[cid],
                                  "frame": record["frame"], "time_s": record["time_s"]})
    np.savez_compressed(args.output / f"{args.name}_source.npz", **source)
    (args.output / f"{args.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
