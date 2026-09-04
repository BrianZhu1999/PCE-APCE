from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_t8_seed2026080601"
OUTPUT_DIR = ROOT / "figures" / "figure4_l96_1024_temporal_psd"

FONT_TITLE = 14
FONT_AXIS = 13
FONT_TICK = 11
FONT_LEGEND = 14

SERIES = [
    ("truth", "Truth", "#E84A3A", "--", 2.55, 10),
    ("aug_enkf", "Aug-EnKF", "#9B59B6", "-", 1.55, 2),
    ("bma", "BMA", "#F39C12", "-", 1.65, 3),
    ("pce", "PCE", "#3775BA", "-", 2.15, 5),
    ("apce", "APCE", "#2ECC71", "-", 2.30, 6),
]


# Figure contract
# Core conclusion: the four assimilation methods preserve different amounts of
# the Lorenz-96 temporal power distribution relative to the same truth.
# Archetype: quantitative spectral comparison.
# Source: one frozen paired seed; five-seed metrics remain the primary
# performance evidence.


def clean_number(value: float, _position: int | None = None) -> str:
    return f"{value:g}"


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": FONT_AXIS,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_fields() -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    with np.load(SOURCE / "shared_asset.npz", allow_pickle=False) as data:
        fields["truth"] = np.asarray(data["truth"], dtype=float)
    for key, filename in {
        "aug_enkf": "aug_enkf_trace.npz",
        "bma": "bma_trace.npz",
        "pce": "pce_trace.npz",
        "apce": "apce_trace.npz",
    }.items():
        with np.load(SOURCE / filename, allow_pickle=False) as data:
            fields[key] = np.asarray(data["mean_states"], dtype=float)
    return fields


def spatial_mean_welch_psd(
    field: np.ndarray,
    *,
    dt: float,
    segment_length: int = 128,
    overlap: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if field.ndim != 2:
        raise ValueError("Expected time-by-state field.")
    if segment_length > field.shape[0]:
        raise ValueError("Welch segment exceeds the available time samples.")
    step = segment_length - overlap
    starts = range(0, field.shape[0] - segment_length + 1, step)
    window = np.hanning(segment_length)
    window_energy = float(np.sum(window**2))
    sample_frequency = 1.0 / dt
    accumulated: list[np.ndarray] = []
    for start in starts:
        segment = field[start : start + segment_length]
        segment = segment - segment.mean(axis=0, keepdims=True)
        transform = np.fft.rfft(segment * window[:, None], axis=0)
        psd = np.abs(transform) ** 2 / (sample_frequency * window_energy)
        if segment_length % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        accumulated.append(psd.mean(axis=1))
    mean_psd = np.mean(np.stack(accumulated, axis=0), axis=0)
    frequencies = np.fft.rfftfreq(segment_length, d=dt)
    return frequencies, mean_psd


def spectral_metrics(frequencies: np.ndarray, truth_psd: np.ndarray, estimate_psd: np.ndarray) -> dict[str, float]:
    valid = frequencies > 0
    log_truth = np.log10(np.maximum(truth_psd[valid], 1.0e-16))
    log_estimate = np.log10(np.maximum(estimate_psd[valid], 1.0e-16))
    log_rmse = float(np.sqrt(np.mean((log_estimate - log_truth) ** 2)))
    truth_total = float(np.trapezoid(truth_psd, frequencies))
    estimate_total = float(np.trapezoid(estimate_psd, frequencies))
    return {
        "log10_psd_rmse": log_rmse,
        "total_power_ratio": estimate_total / max(truth_total, 1.0e-16),
    }


def save_all(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 650},
        ".tiff": {"dpi": 650},
    }.items():
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def draw() -> tuple[list[Path], list[dict[str, str | float]]]:
    fields = load_fields()
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {
        key: spatial_mean_welch_psd(field, dt=0.01) for key, field in fields.items()
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.35))
    for key, label, color, linestyle, linewidth, zorder in SERIES:
        frequencies, psd = spectra[key]
        valid = (frequencies > 0) & (frequencies <= 25)
        ax.plot(
            frequencies[valid],
            psd[valid],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
            label=label,
            zorder=zorder,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Temporal frequency", fontsize=FONT_AXIS)
    ax.set_ylabel("Power spectral density", fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(False)
    ax.legend(
        loc="lower left",
        fontsize=FONT_LEGEND,
        handlelength=2.25,
        borderaxespad=0.30,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.98)
    outputs = save_all(fig, OUTPUT_DIR / "figure4_l96_temporal_psd_four_methods_v1")

    truth_f, truth_psd = spectra["truth"]
    rows: list[dict[str, str | float]] = []
    for key, label, *_ in SERIES[1:]:
        frequencies, psd = spectra[key]
        metrics = spectral_metrics(frequencies, truth_psd, psd)
        rows.append({"method": key, "label": label, **metrics})
    return outputs, rows


def write_source_data(rows: list[dict[str, str | float]]) -> Path:
    path = OUTPUT_DIR / "psd_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "label", "log10_psd_rmse", "total_power_ratio"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    configure_matplotlib()
    outputs, rows = draw()
    source_csv = write_source_data(rows)
    manifest = {
        "case": "Lorenz-96 D=1024",
        "seed": 2026080601,
        "dt": 0.01,
        "welch_segment_length": 128,
        "welch_overlap": 64,
        "spatial_average": "mean over all 1024 state variables",
        "display_frequency_range": [float(1.0 / 1.28), 25.0],
        "source": str(SOURCE),
        "source_data": str(source_csv),
        "outputs": [str(path) for path in outputs],
    }
    (OUTPUT_DIR / "qa_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": manifest["outputs"], "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
