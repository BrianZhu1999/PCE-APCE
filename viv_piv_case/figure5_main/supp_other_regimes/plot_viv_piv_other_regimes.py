"""Render four Figure-5-style supplementary VIV-PIV reconstruction figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.ticker import FuncFormatter, MaxNLocator


HERE = Path(__file__).resolve().parent
FIGURE5_ROOT = HERE.parent
sys.path.insert(0, str(FIGURE5_ROOT / "panels_bcde"))
from plot_bcde_field_strip_gradient_glow import (  # noqa: E402
    _add_group_colorbar,
    _asymmetric_norm,
    _configure,
    _field_axis,
    _symmetric_norm,
)


CASES = ("0463", "0556", "0679", "1359")
UR = {case: int(case) / 100.0 for case in CASES}
DIAMETER_M = 0.05
DPI = 650
FIG_W_PX = 10553
FIG_H_PX = 3900
FIG_W_IN = FIG_W_PX / DPI
FIG_H_IN = FIG_H_PX / DPI


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field_cmap() -> mpl.colors.Colormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "blue_warmwhite_red",
        [
            (0.000, "#185685"),
            (0.125, "#6C93B0"),
            (0.250, "#B6CCD9"),
            (0.375, "#EBEFEE"),
            (0.500, "#F0ECE1"),
            (0.625, "#F8E9D6"),
            (0.750, "#E4B6A7"),
            (0.875, "#BF6B69"),
            (1.000, "#9A1D28"),
        ],
        N=256,
    )


def phase_frames(case, warmup_s: float) -> list[int]:
    warmup = int(np.searchsorted(case.time_s, warmup_s))
    displacement = np.asarray(case.cyl_displ_m, dtype=float)
    centred = displacement - np.median(displacement[warmup:])
    negative = warmup + int(np.argmin(centred[warmup:]))
    positive = warmup + int(np.argmax(centred[warmup:]))
    return sorted((negative, positive))


def load_records(case_id: str, data_root: Path, result_root: Path):
    from hybrid_uncertain_wave.viv_piv_case.io import VIVCase, locate_case

    archive = locate_case(data_root, case_id)
    case = VIVCase.open(archive)
    trace_path = (
        result_root / "runs" / "rank256_stride1" / "traces"
        / f"viv_{case_id}_apce_seed000_layoutadaptive_fullfield_valid_x40y20_ens064_covfull_shr050.npz"
    )
    pod_path = result_root / "models" / "rank256_stride1" / "pod_model.npz"
    with np.load(pod_path, allow_pickle=False) as pod:
        mean = np.asarray(pod["mean"], dtype=np.float32)
        basis = np.asarray(pod["basis"], dtype=np.float32)
    with np.load(trace_path, allow_pickle=False) as trace:
        latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
        trace_time = np.asarray(trace["time_s"], dtype=float)

    records = []
    for frame in phase_frames(case, warmup_s=2.0):
        truth_block, valid_block = case.physical_frames(frame, frame + 1)
        reference = np.asarray(truth_block[0], dtype=np.float32)
        valid = np.asarray(valid_block[0], dtype=bool)
        height, width = reference.shape[:2]
        apce = (mean + latent[frame] @ basis.T).reshape(height, width, 2)
        records.append({
            "time_s": float(trace_time[frame]),
            "frame": int(frame),
            "x": np.asarray(case.x_mm, dtype=float) / (DIAMETER_M * 1000.0),
            "y": np.asarray(case.y_mm, dtype=float) / (DIAMETER_M * 1000.0),
            "valid": valid,
            "reference_u": reference[..., 0],
            "apce_u": apce[..., 0],
            "reference_v": reference[..., 1],
            "apce_v": apce[..., 1],
            "cylinder_y_over_d": float(case.cyl_displ_m[frame] / DIAMETER_M),
        })
    return archive, trace_path, pod_path, records


def add_group_colorbar_lower(fig, axes, norm, cmap, label):
    """Figure-5 colorbar styling with extra clearance below the field axes."""
    fig.canvas.draw()
    left = min(axis.get_position().x0 for axis in axes)
    right = max(axis.get_position().x1 for axis in axes)
    bottom = min(axis.get_position().y0 for axis in axes)
    cbar_bottom = max(0.010, bottom - 0.110)
    cax = fig.add_axes([left, cbar_bottom, right - left, 0.021])
    cax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    cax.patch.set_alpha(0.0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    colorbar.ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    colorbar.ax.patch.set_alpha(0.0)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=11, length=2.5, width=0.6, pad=2)
    colorbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    def format_scaled(value: float, _position: float) -> str:
        scaled = value * 10.0
        if abs(scaled - round(scaled)) < 1.0e-9:
            return str(int(round(scaled)))
        return f"{scaled:.2f}".rstrip("0").rstrip(".")

    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(format_scaled))
    colorbar.set_label(label, fontsize=13, labelpad=2)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)


def render_case(case_id: str, data_root: Path, result_root: Path, output: Path) -> dict[str, object]:
    archive, trace_path, pod_path, records = load_records(case_id, data_root, result_root)
    valid = np.asarray(records[0]["valid"], dtype=bool)
    u_norm = _asymmetric_norm(
        [np.asarray(record[key]) for record in records for key in ("reference_u", "apce_u")],
        valid,
    )
    v_norm = _symmetric_norm(
        [np.asarray(record[key]) for record in records for key in ("reference_v", "apce_v")],
        valid,
    )
    cmap = field_cmap()

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")
    grid = fig.add_gridspec(
        2, 4,
        left=0.040, right=0.990, top=0.895, bottom=0.205,
        wspace=0.20, hspace=0.30,
    )
    axes: list[list[plt.Axes]] = [[], []]
    for row, record in enumerate(records):
        entries = (
            ("reference_u", "Ref.", u_norm),
            ("apce_u", "APCE", u_norm),
            ("reference_v", "Ref.", v_norm),
            ("apce_v", "APCE", v_norm),
        )
        for col, (key, title, norm) in enumerate(entries):
            axis = fig.add_subplot(grid[row, col])
            axes[row].append(axis)
            _field_axis(axis, np.asarray(record[key]), record, norm, cmap)
            axis.set_title(
                f"{title}  ($t={float(record['time_s']):.2f}$ s)",
                fontsize=14,
                pad=7,
                fontweight="normal",
            )
            if row == 0:
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xlabel(r"$x/D$", fontsize=13, labelpad=3)
            if col in (0, 2):
                axis.set_ylabel(r"$y/D$", fontsize=13, labelpad=4)
            else:
                axis.set_ylabel("")

    fig.text(0.018, 0.935, "a", fontsize=22, fontweight="bold", ha="left", va="center")
    fig.text(0.510, 0.935, "b", fontsize=22, fontweight="bold", ha="left", va="center")
    u_axes = [axes[row][col] for row in range(2) for col in (0, 1)]
    v_axes = [axes[row][col] for row in range(2) for col in (2, 3)]
    add_group_colorbar_lower(fig, u_axes, u_norm, cmap, r"$u$ ($\times 10^{-1}$ m s$^{-1}$)")
    add_group_colorbar_lower(fig, v_axes, v_norm, cmap, r"$v$ ($\times 10^{-1}$ m s$^{-1}$)")

    stem = output / f"supp_viv_piv_regime_{case_id}"
    outputs = {
        ".png": stem.with_suffix(".png"),
        ".tiff": stem.with_suffix(".tiff"),
        ".pdf": stem.with_suffix(".pdf"),
        ".svg": stem.with_suffix(".svg"),
    }
    fig.savefig(outputs[".png"], dpi=DPI, facecolor="white", pad_inches=0)
    fig.savefig(outputs[".tiff"], dpi=DPI, facecolor="white", pad_inches=0, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outputs[".pdf"], facecolor="white", pad_inches=0)
    fig.savefig(outputs[".svg"], facecolor="white", pad_inches=0)
    plt.close(fig)

    with Image.open(outputs[".png"]) as image:
        png_size = list(image.size)
    return {
        "case_id": case_id,
        "reduced_velocity": UR[case_id],
        "layout": "two phase rows by four columns: Ref u, APCE u, Ref v, APCE v",
        "selection": "negative and positive extrema of median-centred cylinder displacement after 2 s warm-up; ordered by time",
        "frames": [record["frame"] for record in records],
        "times_s": [record["time_s"] for record in records],
        "cylinder_y_over_d": [record["cylinder_y_over_d"] for record in records],
        "true_coordinate_aspect": True,
        "cylinder_radius_over_d": 0.5,
        "color_limits": {
            "u": {"vmin": float(u_norm.vmin), "vmax": float(u_norm.vmax)},
            "v": {"vmin": float(v_norm.vmin), "vmax": float(v_norm.vmax)},
            "scope": "shared across both displayed phases and Ref/APCE within this regime",
        },
        "sources": {
            "truth_archive": str(archive),
            "apce_trace": str(trace_path),
            "pod_model": str(pod_path),
        },
        "outputs": {suffix: str(path) for suffix, path in outputs.items()},
        "output_sha256": {suffix: sha256_file(path) for suffix, path in outputs.items()},
        "png_size": png_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root))
    _configure()
    args.output.mkdir(parents=True, exist_ok=True)

    figures = [render_case(case, args.data_root, args.result_root, args.output) for case in CASES]
    registry_path = args.output / "supp_viv_piv_other_regimes_panel_registry.csv"
    rows = []
    for item in figures:
        rows.append({
            "figure": f"supp_viv_piv_regime_{item['case_id']}",
            "case_id": item["case_id"],
            "reduced_velocity": item["reduced_velocity"],
            "frames": ";".join(str(v) for v in item["frames"]),
            "times_s": ";".join(f"{v:.6f}" for v in item["times_s"]),
            "selection": item["selection"],
            "truth_source": item["sources"]["truth_archive"],
            "apce_trace_source": item["sources"]["apce_trace"],
            "source_script": str(Path(__file__).resolve()),
        })
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "figure_set": "supp_viv_piv_other_regimes",
        "backend": "Python/matplotlib only",
        "visual_source": str(FIGURE5_ROOT / "panels_bcde" / "plot_bcde_field_strip_gradient_glow.py"),
        "visible_text_policy": "panel letters, Ref./APCE titles, times, axes, ticks and colorbar labels only",
        "figures": figures,
        "panel_registry": str(registry_path),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    metadata_path = args.output / "supp_viv_piv_other_regimes_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
