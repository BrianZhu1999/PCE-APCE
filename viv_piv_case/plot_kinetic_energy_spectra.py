"""Plot full-field kinetic-energy spectra for the VIV-PIV reconstruction runs."""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib as mpl
import numpy as np
import torch

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
    "legend.frameon": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


RUNS = {
    "PCE, diagonal R": ("pce", False, "#4C78A8", "--"),
    "APCE, diagonal R": ("apce", False, "#F28E2B", "--"),
    "PCE, full R": ("pce", True, "#1F4E79", "-"),
    "APCE, full R": ("apce", True, "#C44E52", "-"),
}


def load_latent(run_root: pathlib.Path, case_id: str, method: str, full: bool) -> np.ndarray:
    suffix = "_layout20x40_covfull" if full else "_layout20x40"
    path = run_root / "traces" / f"viv_{case_id}_{method}_seed000{suffix}.npz"
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["latent_estimate"], dtype=np.float32)


def kinetic_energy_series(
    case: VIVCase,
    pod: PODModel,
    latents: dict[str, np.ndarray],
    device: torch.device,
    block: int = 32,
) -> dict[str, np.ndarray]:
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    output = {"truth": np.empty(case.time_s.size, dtype=np.float64)}
    output.update({name: np.empty(case.time_s.size, dtype=np.float64) for name in latents})
    with torch.inference_mode():
        for start, values, valid in case.iter_physical(block=block):
            stop = start + values.shape[0]
            truth = torch.as_tensor(values, dtype=torch.float32, device=device)
            valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
            truth_field = truth.reshape(truth.shape[0], -1, 2)
            valid_pixel = valid_t.reshape(valid_t.shape[0], -1, 2)[..., 0]
            output["truth"][start:stop] = (
                0.5 * torch.sum(torch.sum(truth_field.square(), dim=2) * valid_pixel, dim=1)
                / valid_pixel.sum(dim=1).clamp_min(1)
            ).cpu().numpy()
            latent_batch = torch.as_tensor(
                np.stack([latents[name][start:stop] for name in latents]),
                dtype=torch.float32,
                device=device,
            )
            predictions = mean[None, None, :] + torch.matmul(latent_batch, basis.mT)
            predictions = predictions.reshape(len(latents), stop - start, -1, 2)
            valid_for_prediction = valid_pixel[None, :, :]
            energy = 0.5 * torch.sum(
                torch.sum(predictions.square(), dim=3) * valid_for_prediction,
                dim=2,
            ) / valid_for_prediction.sum(dim=2).clamp_min(1)
            for index, name in enumerate(latents):
                output[name][start:stop] = energy[index].cpu().numpy()
    return output


def spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    centred = signal - signal.mean()
    window = np.hanning(signal.size)
    transformed = np.fft.rfft(centred * window)
    frequencies = np.fft.rfftfreq(signal.size, dt)
    power = dt * np.abs(transformed) ** 2 / np.sum(window**2)
    if power.size > 2:
        power[1:-1] *= 2.0
    return frequencies, power / max(np.trapz(power, frequencies), 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VIV-PIV kinetic-energy spectra.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    variant = args.variant or f"rank{int(config['rank'])}_stride1"
    model_root = pathlib.Path(config["output_root"]) / "models" / variant
    run_root = pathlib.Path(config["output_root"]) / "runs" / variant
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pod = PODModel.load(model_root / "pod_model.npz")
    paths = list_cases(pathlib.Path(config["data_root"]))
    all_series: dict[str, dict[str, np.ndarray]] = {}
    for case_id in config["test_cases"]:
        case_id = str(case_id)
        case = VIVCase.open(paths[case_id])
        latent = {
            name: load_latent(run_root, case_id, method, full)
            for name, (method, full, _color, _line) in RUNS.items()
        }
        all_series[case_id] = kinetic_energy_series(case, pod, latent, device)

    distances: dict[str, list[float]] = {name: [] for name in RUNS}
    peak_errors: dict[str, list[float]] = {name: [] for name in RUNS}
    spectra: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for case_id, series in all_series.items():
        spectra[case_id] = {}
        frequencies, truth_power = spectrum(series["truth"], 0.1)
        spectra[case_id]["truth"] = (frequencies, truth_power)
        peak_band = (frequencies >= 0.05) & (frequencies <= 2.0)
        truth_peak = frequencies[peak_band][np.argmax(truth_power[peak_band])]
        for name in RUNS:
            f, power = spectrum(series[name], 0.1)
            spectra[case_id][name] = (f, power)
            distances[name].append(float(np.trapz(np.abs(power - truth_power), f)))
            peak = f[peak_band][np.argmax(power[peak_band])]
            peak_errors[name].append(abs(float(peak - truth_peak)) / max(float(truth_peak), 1e-12))

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), gridspec_kw={"wspace": 0.28, "hspace": 0.38})
    axes_flat = axes.ravel()
    for axis, case_id in zip(axes_flat[:5], config["test_cases"]):
        case_id = str(case_id)
        f, truth_power = spectra[case_id]["truth"]
        axis.loglog(f[1:], truth_power[1:], color="#222222", lw=1.35, label="truth")
        for name, (_method, _full, color, line) in RUNS.items():
            f, power = spectra[case_id][name]
            axis.loglog(f[1:], power[1:], color=color, lw=1.05, ls=line, label=name)
        axis.set_xlim(0.01, 5.0)
        axis.set_title(rf"$U_r={int(case_id)/100:.2f}$", loc="left", fontweight="bold")
        axis.set_xlabel("frequency (Hz)")
        axis.set_ylabel("normalised PSD")
        axis.grid(True, which="major", color="#D9D9D9", lw=0.45, alpha=0.65)
    axis = axes_flat[5]
    names = list(RUNS)
    means = [np.mean(distances[name]) for name in names]
    errors = [np.std(distances[name], ddof=1) / np.sqrt(len(config["test_cases"])) for name in names]
    xpos = np.arange(len(names))
    axis.bar(xpos, means, yerr=errors, color=[RUNS[name][2] for name in names], width=0.72, capsize=2.0, edgecolor="none")
    axis.set_xticks(xpos, ["PCE\ndiag", "APCE\ndiag", "PCE\nfull", "APCE\nfull"], rotation=0)
    axis.set_ylabel("spectral L1 distance")
    axis.set_title("Five-case spectral discrepancy", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.65)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=5, fontsize=6.7, handlelength=2.2)
    fig.suptitle("VIV-PIV kinetic-energy spectra under sparse-field assimilation", y=1.055, fontsize=10, fontweight="bold")
    fig.savefig(output / "kinetic_energy_spectra_comparison.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "kinetic_energy_spectra_comparison.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output / "kinetic_energy_spectra_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    np.savez_compressed(
        output / "kinetic_energy_spectra_source.npz",
        **{f"{case}_{name}": values for case, series in all_series.items() for name, values in series.items()},
    )
    summary = {
        "figure": "kinetic_energy_spectra_comparison",
        "cases": list(config["test_cases"]),
        "dt_s": 0.1,
        "spectral_distance": distances,
        "relative_peak_frequency_error": peak_errors,
        "device": str(device),
    }
    (output / "kinetic_energy_spectra_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output / "kinetic_energy_spectra_comparison.png")


if __name__ == "__main__":
    main()
