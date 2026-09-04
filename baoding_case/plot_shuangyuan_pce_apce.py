#!/usr/bin/env python3
"""Plot per-seed PCE/APCE 3-D trajectories for shuangyuan_4.

The output is an audit/inspection bundle, not a manuscript figure.  Coordinates
are the already-centered local ENU coordinates from the target frontend and
are displayed in kilometres to avoid misleading scientific-notation axes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"truth": "#20262C", "pce": "#1D6F8A", "apce": "#C75A3C"}
SEEDS = (2026081900, 2026081901, 2026081902, 2026081903, 2026081904)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def xyz(rows: list[dict], prefix: str = "") -> np.ndarray:
    names = [f"{prefix}{name}" for name in ("px", "py", "pz")]
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def save_source(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def equal_limits(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_xyz = np.concatenate(arrays, axis=0) / 1000.0
    lo = all_xyz.min(axis=0)
    hi = all_xyz.max(axis=0)
    center = 0.5 * (lo + hi)
    half = max(float(np.max(hi - lo)) * 0.55, 0.05)
    return center - half, center + half


def plot_one(
    target: int,
    seed: int,
    result_root: Path,
    output: Path,
) -> dict:
    target_root = result_root / f"target{target}"
    truth_rows = read_csv(target_root / "frontend" / "gps_truth.csv")
    pce_payload = read_json(target_root / "runs" / f"pce_seed_{seed}.json")
    apce_payload = read_json(target_root / "runs" / f"apce_seed_{seed}.json")
    pce_rows = pce_payload["records"]
    apce_rows = apce_payload["records"]
    if len(truth_rows) != len(pce_rows) or len(pce_rows) != len(apce_rows):
        raise RuntimeError(f"length mismatch target={target} seed={seed}")
    truth = np.asarray([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in truth_rows])
    pce = xyz(pce_rows)
    apce = xyz(apce_rows)
    # Already local ENU coordinates; only convert metres to kilometres for axes.
    truth_km, pce_km, apce_km = truth / 1000.0, pce / 1000.0, apce / 1000.0
    lower, upper = equal_limits([truth, pce, apce])

    fig = plt.figure(figsize=(9.2, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth_km[:, 0], truth_km[:, 1], truth_km[:, 2], color=COLORS["truth"], lw=2.0, label="GPS truth")
    ax.plot(pce_km[:, 0], pce_km[:, 1], pce_km[:, 2], color=COLORS["pce"], lw=1.15, label="PCE")
    ax.plot(apce_km[:, 0], apce_km[:, 1], apce_km[:, 2], color=COLORS["apce"], lw=1.15, label="APCE")
    ax.scatter(*truth_km[0], color=COLORS["truth"], s=20, marker="o", depthshade=False)
    ax.scatter(*truth_km[-1], color=COLORS["truth"], s=24, marker="s", depthshade=False)
    ax.set_xlim(lower[0], upper[0]); ax.set_ylim(lower[1], upper[1]); ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-55)
    ax.set_xlabel("East (km)", labelpad=6)
    ax.set_ylabel("North (km)", labelpad=6)
    ax.set_zlabel("Up (km)", labelpad=6)
    ax.set_title(f"shuangyuan_4 — target {target}, seed {seed}", loc="left", weight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(
        0.02,
        0.01,
        "Local ENU coordinates; axes shown in km. Circle=start, square=end.",
        fontsize=7,
        color="#59636B",
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"shuangyuan4_target{target}_pce_apce_3d_seed_{seed}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    source_rows = []
    for truth_row, pce_row, apce_row in zip(truth_rows, pce_rows, apce_rows):
        source_rows.append(
            {
                "target": target,
                "seed": seed,
                "time_s": truth_row["time_s"],
                "truth_east_m": truth_row["px"],
                "truth_north_m": truth_row["py"],
                "truth_up_m": truth_row["pz"],
                "pce_east_m": pce_row["px"],
                "pce_north_m": pce_row["py"],
                "pce_up_m": pce_row["pz"],
                "apce_east_m": apce_row["px"],
                "apce_north_m": apce_row["py"],
                "apce_up_m": apce_row["pz"],
                "pce_error_m": pce_row["position_error_m"],
                "apce_error_m": apce_row["position_error_m"],
            }
        )
    source = stem.with_name(stem.name + "_source.csv")
    save_source(source, source_rows)
    return {
        "target": target,
        "seed": seed,
        "figure_png": str(stem.with_suffix(".png")),
        "figure_pdf": str(stem.with_suffix(".pdf")),
        "figure_svg": str(stem.with_suffix(".svg")),
        "source_data": str(source),
        "frames": len(truth_rows),
        "pce_rmse_m": float(np.sqrt(np.mean(np.asarray([float(r["position_error_m"]) for r in pce_rows]) ** 2))),
        "apce_rmse_m": float(np.sqrt(np.mean(np.asarray([float(r["position_error_m"]) for r in apce_rows]) ** 2))),
        "coordinate_system": "centered local ENU; displayed in km",
    }


def plot_overview(target: int, entries: list[dict], output: Path) -> dict:
    fig = plt.figure(figsize=(12.0, 7.8), constrained_layout=True)
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(5)]
    axes.append(fig.add_subplot(2, 3, 6))
    rmses = []
    for ax, entry in zip(axes[:5], entries):
        t = entry["seed"]
        target_root = Path(entry["figure_png"]).parent.parent.parent / f"target{target}"
        # The entry carries source data; use it to keep the overview traceable.
        rows = read_csv(Path(entry["source_data"]))
        truth = np.asarray([[float(r["truth_east_m"]), float(r["truth_north_m"]), float(r["truth_up_m"])] for r in rows]) / 1000
        pce = np.asarray([[float(r["pce_east_m"]), float(r["pce_north_m"]), float(r["pce_up_m"])] for r in rows]) / 1000
        apce = np.asarray([[float(r["apce_east_m"]), float(r["apce_north_m"]), float(r["apce_up_m"])] for r in rows]) / 1000
        ax.plot(*truth.T, color=COLORS["truth"], lw=1.4)
        ax.plot(*pce.T, color=COLORS["pce"], lw=0.8)
        ax.plot(*apce.T, color=COLORS["apce"], lw=0.8)
        lo, hi = equal_limits([truth * 1000, pce * 1000, apce * 1000])
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-55)
        ax.set_title(str(t), fontsize=8)
        ax.set_xlabel("E (km)", labelpad=1); ax.set_ylabel("N (km)", labelpad=1); ax.set_zlabel("U (km)", labelpad=1)
        ax.tick_params(labelsize=6)
        rmses.append((entry["seed"], entry["pce_rmse_m"], entry["apce_rmse_m"]))
    ax = axes[5]
    ax.axis("off")
    ax.text(0.0, 1.0, f"Target {target}\nper-seed 3-D trajectories", va="top", weight="bold", fontsize=11)
    ax.text(0.0, 0.82, "All coordinates: centered local ENU\nAxes in kilometres", va="top", fontsize=8)
    y = 0.62
    for seed, pce_rmse, apce_rmse in rmses:
        ax.text(0.0, y, f"{seed}: PCE {pce_rmse:.0f} m | APCE {apce_rmse:.0f} m", fontsize=8)
        y -= 0.10
    ax.plot([], [], color=COLORS["truth"], lw=2, label="GPS truth")
    ax.plot([], [], color=COLORS["pce"], lw=1.2, label="PCE")
    ax.plot([], [], color=COLORS["apce"], lw=1.2, label="APCE")
    ax.legend(loc="lower left", fontsize=8)
    stem = output / f"shuangyuan4_target{target}_pce_apce_3d_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return {"target": target, "figure_png": str(stem.with_suffix(".png")), "figure_pdf": str(stem.with_suffix(".pdf"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_style()
    registry = []
    for target in (1, 2):
        entries = [plot_one(target, seed, args.result_root, args.output / f"target{target}") for seed in SEEDS]
        registry.extend(entries)
        plot_overview(target, entries, args.output / f"target{target}")
    payload = {
        "claim": "Inspection-only visualization of the admitted dual-target real-data smoke; no superiority claim.",
        "backend": "python/matplotlib",
        "source_result": str(args.result_root),
        "coordinate_system": "centered local ENU; all axes displayed in km",
        "entries": registry,
    }
    (args.output / "figure_registry.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
