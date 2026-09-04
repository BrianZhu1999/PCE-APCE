from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_t8_seed2026080601"
OUTPUT_DIR = ROOT / "figures" / "figure4_l96_1024_timeseries_grid"

FONT_TITLE = 14
FONT_ROW = 12
FONT_AXIS = 13
FONT_TICK = 11

METHODS = [
    ("truth", "GT", "#E84A3A", 1.55),
    ("apce", "APCE", "#2ECC71", 1.35),
    ("pce", "PCE", "#3775BA", 1.35),
    ("bma", "BMA", "#F39C12", 1.35),
    ("aug_enkf", "Aug-EnKF", "#9B59B6", 1.35),
]

# Five fixed, evenly separated ring states. The last point is a measured
# sensor (1016 = 127*8), so all rows remain tied to the actual geometry.
STATE_INDICES = np.asarray([0, 256, 512, 768, 1016], dtype=int)


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": FONT_AXIS,
            "axes.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "xtick.bottom": False,
            "ytick.left": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["times"], dtype=float), np.asarray(data["mean_states"], dtype=float)


def save_all(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 650},
        ".tiff": {"dpi": 650},
    }.items():
        target = base.with_suffix(suffix)
        fig.savefig(target, bbox_inches="tight", pad_inches=0.025, **kwargs)
        paths.append(target)
    plt.close(fig)
    return paths


def draw_grid() -> list[Path]:
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, _, _, _ in METHODS:
        path = SOURCE / ("apce_trace.npz" if key == "apce" else "pce_trace.npz" if key == "pce" else "bma_trace.npz" if key == "bma" else "aug_enkf_trace.npz" if key == "aug_enkf" else "shared_asset.npz")
        with np.load(path, allow_pickle=False) as data:
            if key == "truth":
                traces[key] = (np.arange(data["truth"].shape[0], dtype=float) * 0.01, np.asarray(data["truth"], dtype=float))
            else:
                traces[key] = (np.asarray(data["times"], dtype=float), np.asarray(data["mean_states"], dtype=float))

    fig, axes = plt.subplots(
        nrows=len(STATE_INDICES),
        ncols=len(METHODS),
        figsize=(10.8, 6.15),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.060, right=0.995, bottom=0.075, top=0.925, wspace=0.050, hspace=0.30)

    for col, (_, title, _, _) in enumerate(METHODS):
        axes[0, col].set_title(title, fontsize=FONT_TITLE, pad=8)

    for row, state_index in enumerate(STATE_INDICES):
        series = [traces[key][1][:, state_index] for key, _, _, _ in METHODS]
        row_min = float(min(np.min(values) for values in series))
        row_max = float(max(np.max(values) for values in series))
        margin = max(0.08, 0.075 * (row_max - row_min))
        limits = (row_min - margin, row_max + margin)
        axes[row, 0].text(
            -0.14,
            0.50,
            f"$x_{{{state_index}}}$",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FONT_ROW,
        )
        for col, (key, _, color, linewidth) in enumerate(METHODS):
            ax = axes[row, col]
            times, _ = traces[key]
            ax.plot(times, traces[key][1][:, state_index], color=color, lw=linewidth, solid_capstyle="round")
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(*limits)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_color("#C8CBCD")
                spine.set_linewidth(0.7)
        if row == len(STATE_INDICES) - 1:
            axes[row, 2].set_xlabel("$t$", fontsize=FONT_AXIS, labelpad=7)

    outputs = save_all(fig, OUTPUT_DIR / "figure4_l96_1024_obs128_t8_timeseries_grid_v1")
    (OUTPUT_DIR / "qa_manifest.json").write_text(
        json.dumps(
            {
                "layout": "5 rows x 5 columns",
                "rows": [int(value) for value in STATE_INDICES],
                "columns": [title for _, title, _, _ in METHODS],
                "seed": 2026080601,
                "shared_y_limits": "within each row",
                "source": str(SOURCE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    configure_matplotlib()
    outputs = draw_grid()
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
