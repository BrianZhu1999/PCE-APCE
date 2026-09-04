#!/usr/bin/env python3
"""Plot the Baoding single-source task, node geometry and 19-mic manifold.

All inputs are read from the authoritative remote frontend bundle when the
script is run on Super-Server.  The plotted array geometry is the explicit
geometry used by the current Python MUSIC frontend: three orthogonal linear
subarrays (x/y/z), not an undocumented claim about private author code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


CHANNEL_GROUPS = {
    "x-arm": (9, 8, 7, 1, 2, 3),
    "y-arm": (12, 11, 10, 4, 5, 6),
    "z-arm": (19, 18, 17, 13, 14, 15, 16),
}
GROUP_COLORS = {"x-arm": "#C65B3C", "y-arm": "#1E7A70", "z-arm": "#3E6AA8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hms(seconds: float) -> str:
    seconds = int(round(seconds)) % 86400
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def array_geometry(spacing: float) -> dict[str, list[tuple[int, float, float, float]]]:
    horizontal = [-3.0 * spacing, -2.0 * spacing, -spacing, spacing, 2.0 * spacing, 3.0 * spacing]
    vertical = [-2.13 * spacing, -1.53 * spacing, -0.93 * spacing, 0.0, 1.0 * spacing, 2.0 * spacing, 3.0 * spacing]
    positions: dict[str, list[tuple[int, float, float, float]]] = {}
    for name, channels in CHANNEL_GROUPS.items():
        if name == "x-arm":
            positions[name] = [(channel, coordinate, 0.0, 0.0) for channel, coordinate in zip(channels, horizontal)]
        elif name == "y-arm":
            positions[name] = [(channel, 0.0, coordinate, 0.0) for channel, coordinate in zip(channels, horizontal)]
        else:
            positions[name] = [(channel, 0.0, 0.0, coordinate) for channel, coordinate in zip(channels, vertical)]
    return positions


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame_summary(path: Path) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, set[float]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row["segment"], set()).add(float(row["time_s"]))
    output: dict[str, dict[str, float | int]] = {}
    for segment, values in grouped.items():
        times = sorted(values)
        output[segment] = {"frames": len(times), "start_s": times[0], "end_s": times[-1], "elapsed_s": times[-1] - times[0]}
    return output


def local_nodes(manifest: dict) -> tuple[dict[int, dict[str, float]], list[float]]:
    values = {int(node): {key: float(value[key]) for key in ("x", "y", "z")} for node, value in manifest["nodes"].items()}
    center = [sum(item[key] for item in values.values()) / len(values) for key in ("x", "y", "z")]
    local = {node: {key: values[node][key] - center[index] for index, key in enumerate(("x", "y", "z"))} for node in values}
    return local, center


def save_node_csv(path: Path, manifest: dict, local: dict[int, dict[str, float]], center: list[float]) -> None:
    fields = ["node_id", "ip_suffix", "global_x_utm_m", "global_y_utm_m", "global_z_m", "local_E_m", "local_N_m", "local_U_m"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for node in sorted(local):
            raw = manifest["nodes"][str(node)]
            writer.writerow({
                "node_id": node,
                "ip_suffix": raw.get("ip"),
                "global_x_utm_m": raw["x"],
                "global_y_utm_m": raw["y"],
                "global_z_m": raw["z"],
                "local_E_m": local[node]["x"],
                "local_N_m": local[node]["y"],
                "local_U_m": local[node]["z"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cartesian_manifest = read_manifest(args.frontend / "frontend_manifest.json")
    source_frontend = Path(cartesian_manifest.get("source_frontend", str(args.frontend)))
    source_manifest = read_manifest(source_frontend / "frontend_manifest.json")
    frame_summary = read_frame_summary(args.frontend / "observations_cartesian.csv")
    local, center = local_nodes(source_manifest)
    positions = array_geometry(args.spacing_m)
    save_node_csv(args.output / "baoding_node_coordinates.csv", source_manifest, local, center)

    # Figure contract: the figure establishes the physical scale and the
    # exact observation protocol before any PCE/APCE error is interpreted.
    fig = plt.figure(figsize=(10.2, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.06, 0.94), height_ratios=(1.0, 1.0))
    ax_map = fig.add_subplot(grid[0, 0])
    ax_global = fig.add_subplot(grid[0, 1])
    ax_array = fig.add_subplot(grid[1, 0], projection="3d")
    ax_protocol = fig.add_subplot(grid[1, 1])

    # A: local ENU node map.
    east = [local[node]["x"] for node in sorted(local)]
    north = [local[node]["y"] for node in sorted(local)]
    elevation = [local[node]["z"] for node in sorted(local)]
    scatter = ax_map.scatter(east, north, c=elevation, cmap="viridis", s=74, edgecolor="white", linewidth=0.7, zorder=3)
    for node in sorted(local):
        x, y = local[node]["x"], local[node]["y"]
        ax_map.annotate(f"N{node}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, fontweight="bold")
    ax_map.scatter([0], [0], marker="+", color="#222222", s=60, linewidth=1.1, label="array-centre origin")
    ax_map.set_xlabel("Local East (m)")
    ax_map.set_ylabel("Local North (m)")
    ax_map.set_title("A  Nine acoustic nodes in local ENU", loc="left", fontweight="bold")
    ax_map.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.03)
    cbar.set_label("Local Up (m)")
    ax_map.legend(loc="lower right", fontsize=6.5)

    # B: original global UTM coordinates, with a centred inset-like display.
    global_x = [float(source_manifest["nodes"][str(node)]["x"]) for node in sorted(local)]
    global_y = [float(source_manifest["nodes"][str(node)]["y"]) for node in sorted(local)]
    ax_global.scatter(global_x, global_y, c=elevation, cmap="viridis", s=74, edgecolor="white", linewidth=0.7)
    for node, x, y in zip(sorted(local), global_x, global_y):
        ax_global.annotate(f"N{node}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, fontweight="bold")
    ax_global.set_xlabel("Global X / UTM-like coordinate (m)")
    ax_global.set_ylabel("Global Y / UTM-like coordinate (m)")
    ax_global.set_title("B  Original coordinate scale", loc="left", fontweight="bold")
    ax_global.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax_global.tick_params(axis="x", labelrotation=25)
    ax_global.text(0.02, 0.02, f"centre = ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m\nlocal plot subtracts this centre", transform=ax_global.transAxes, fontsize=6.3, va="bottom", bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85, "pad": 3})

    # C: explicit 19-mic array manifold used by the current MUSIC frontend.
    for group, rows in positions.items():
        xyz = [(row[1], row[2], row[3]) for row in rows]
        ax_array.plot([p[0] for p in xyz], [p[1] for p in xyz], [p[2] for p in xyz], color=GROUP_COLORS[group], lw=1.0, alpha=0.85)
        ax_array.scatter([p[0] for p in xyz], [p[1] for p in xyz], [p[2] for p in xyz], color=GROUP_COLORS[group], s=28, depthshade=False, label=f"{group} ({len(rows)} mics)")
        for channel, x, y, z in rows:
            ax_array.text(x, y, z, str(channel), fontsize=6.1, color=GROUP_COLORS[group])
    ax_array.scatter([0], [0], [0], marker="+", color="#222222", s=45, linewidth=1.0)
    ax_array.set_xlabel("x (m)", labelpad=1)
    ax_array.set_ylabel("y (m)", labelpad=1)
    ax_array.set_zlabel("z (m)", labelpad=1)
    ax_array.set_title("C  19-microphone 3-D array manifold", loc="left", fontweight="bold", pad=8)
    ax_array.set_box_aspect((1.0, 1.0, 0.75))
    ax_array.legend(loc="upper left", fontsize=6.5, bbox_to_anchor=(-0.02, 1.03))
    ax_array.view_init(elev=22, azim=35)

    # D: exact task protocol and the distinction between raw and admitted frames.
    ax_protocol.axis("off")
    cal = frame_summary["danyuan_panxuan_2"]
    eva = frame_summary["danyuan_panxuan_3"]
    nominal = source_manifest.get("config", {})
    lines = [
        "D  Task protocol",
        "",
        f"Calibration:  {cal['frames']} frames, {cal['elapsed_s']:.0f} s represented",
        f"             {hms(cal['start_s'])} - {hms(cal['end_s'])}",
        f"Evaluation:   {eva['frames']} frames, {eva['elapsed_s']:.0f} s represented",
        f"             {hms(eva['start_s'])} - {hms(eva['end_s'])}",
        "             250/267 Cartesian frames admitted",
        "",
        "Spatial frontend",
        "  9 nodes x 19 microphones = 171 channels",
        "  3 orthogonal nonuniform linear subarrays: 6 + 6 + 7",
        f"  nominal spacing s = {args.spacing_m:.2f} m",
        "  one DOA pair per node per 1 s update",
        "",
        "Current MUSIC implementation",
        f"  sample rate: {nominal.get('sample_rate_expected', 3050)} Hz",
        f"  centre/band: {nominal.get('fc_hz', 300)} / {nominal.get('bandwidth_hz', 400)} Hz",
        f"  NFFT / snapshots: {nominal.get('nfft', 128)} / {nominal.get('snapshots', 25)}",
        "  Cartesian state: [E,N,U,vE,vN,vU]",
        "",
        "GPS role: calibration and offline evaluation only",
    ]
    ax_protocol.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.2, linespacing=1.35)

    fig.suptitle("Baoding single-source acoustic task: time, spatial geometry and array manifold", x=0.02, ha="left", fontsize=12, fontweight="bold")
    stem = args.output / "baoding_single_source_task_geometry"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

    registry = {
        "figure": str(stem),
        "backend": "Python/matplotlib",
        "core_conclusion": "The evaluated task is a 267-frame, 1-s-update, single-source experiment using nine spatially distributed 19-microphone nodes and an explicit three-arm Cartesian array manifold.",
        "panels": {
            "A_local_nodes": "node coordinates translated by arithmetic mean of the nine global node positions",
            "B_global_nodes": "original node coordinates from frontend_manifest.json",
            "C_array_manifold": "explicit x/y/z subarray coordinates from run_baoding.py with spacing 0.50 m",
            "D_protocol": "frame counts, times, channel counts and current MUSIC settings",
        },
        "cartesian_manifest_sha256": sha256(args.frontend / "frontend_manifest.json"),
        "source_manifest_sha256": sha256(source_frontend / "frontend_manifest.json"),
        "source_observations_sha256": sha256(args.frontend / "observations_cartesian.csv"),
        "array_geometry": {"channel_groups": CHANNEL_GROUPS, "spacing_m": args.spacing_m, "positions": positions},
        "gps_role": "calibration and offline evaluation only",
    }
    (args.output / "baoding_single_source_task_geometry_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frame_summary": frame_summary, "center_xyz_global": center, "node_coordinates": local, "array_geometry": positions}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
