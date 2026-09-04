#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

COLORS = {"DEnKF": "#777777", "BMA": "#3E7C82", "PCE": "#3775BA", "APCE": "#E28E2C"}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frf", type=Path, required=True)
    parser.add_argument("--path-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--representative", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    args = parser.parse_args()
    frf = np.load(args.frf)
    path = json.loads(args.path_json.read_text(encoding="utf-8"))
    summary = read_summary(args.summary)
    representative = np.load(args.representative)
    admission = json.loads(args.admission.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), facecolor="white")
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    source_rows = []

    frequency = frf["frequency"]
    h1 = frf["h1"]
    mask = (frequency >= 6.0) & (frequency <= 8.5)
    for channel in range(3):
        magnitude = 20.0 * np.log10(np.maximum(np.abs(h1[:, channel]), 1e-12))
        ax_a.plot(frequency[mask], magnitude[mask], lw=1.0, label=f"Acceleration {channel + 1}")
        for x, value in zip(frequency[mask], magnitude[mask]):
            source_rows.append({"panel": "a", "series": f"Acceleration {channel + 1}", "x": x, "value": value})
    ax_a.axvline(7.3, color="#AA3333", lw=0.8, ls="--")
    ax_a.set_xlabel("Frequency (Hz)")
    ax_a.set_ylabel("FRF magnitude (dB)")
    ax_a.set_title("Level 1 response near wing torsion", fontsize=8, loc="left")
    ax_a.legend(fontsize=5.8, ncol=1)
    ax_a.grid(axis="y", color="#E9E9E9", lw=0.45)

    candidates = path["coarse_candidate_audit"]
    alpha = np.asarray([row["alpha"] for row in candidates])
    damping = np.asarray([row["damping_log_slope"] for row in candidates])
    ax_b.plot(alpha, damping, color="#3775BA", marker="o", ms=3.5, lw=1.1)
    ax_b.axhline(float(path["mean"][1]), color="#777777", lw=0.7, ls="--")
    ax_b.set_xlabel(r"Candidate coordinate $\alpha$")
    ax_b.set_ylabel("Log-damping slope")
    ax_b.set_title("Bootstrap modal-uncertainty path", fontsize=8, loc="left")
    ax_b.grid(axis="y", color="#E9E9E9", lw=0.45)
    for x, value in zip(alpha, damping):
        source_rows.append({"panel": "b", "series": "candidate", "x": x, "value": value})

    time = representative["time"]
    truth = representative["truth"][:, 2]
    estimate = representative["mean"][:, 2]
    display = (time >= 20.0) & (time <= 40.0)
    ax_c.plot(time[display], truth[display], color="#222222", lw=1.1, label="Measured")
    ax_c.plot(time[display], estimate[display], color=COLORS["APCE"], lw=1.0, label="APCE")
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("Held-out acceleration")
    ax_c.set_title("Level 4 held-out response", fontsize=8, loc="left")
    ax_c.legend(fontsize=6)
    ax_c.grid(axis="y", color="#E9E9E9", lw=0.45)
    for x, target, prediction in zip(time[display], truth[display], estimate[display]):
        source_rows.append({"panel": "c", "series": "Measured", "x": x, "value": target})
        source_rows.append({"panel": "c", "series": "APCE", "x": x, "value": prediction})

    levels = (2, 4, 6)
    methods = ("DEnKF", "BMA", "PCE", "APCE")
    for axis, condition, panel in ((ax_d, "standard", "d"), (ax_e, "blackout", "e")):
        for method in methods:
            values = []
            for level in levels:
                row = next(item for item in summary if int(item["level"]) == level and item["condition"] == condition and item["method"] == method)
                values.append(float(row["mean_heldout_nrmse"]))
                source_rows.append({"panel": panel, "series": method, "x": level, "value": values[-1]})
            axis.plot(levels, values, marker="o", ms=3.4, lw=1.0, color=COLORS[method], label=method)
        axis.axhline(0.5, color="#AA3333", lw=0.7, ls="--")
        axis.set_xticks(levels)
        axis.set_xlabel("Validation level")
        axis.set_ylabel("Held-out nRMSE")
        axis.set_title(f"{condition.capitalize()} validation", fontsize=8, loc="left")
        axis.grid(axis="y", color="#E9E9E9", lw=0.45)
    ax_d.legend(fontsize=5.8, ncol=2)

    fixed = []
    oracle = []
    for level in levels:
        denkf = next(item for item in summary if int(item["level"]) == level and item["condition"] == "standard" and item["method"] == "DEnKF")
        pce = next(item for item in summary if int(item["level"]) == level and item["condition"] == "standard" and item["method"] == "PCE")
        fixed.append(float(denkf["mean_heldout_nrmse"]))
        oracle.append(float(pce["mean_oracle_candidate_nrmse"]))
        source_rows.append({"panel": "f", "series": "Fixed DEnKF", "x": level, "value": fixed[-1]})
        source_rows.append({"panel": "f", "series": "Candidate oracle", "x": level, "value": oracle[-1]})
    x = np.arange(len(levels))
    ax_f.bar(x - 0.16, fixed, width=0.32, color="#777777", label="Fixed DEnKF")
    ax_f.bar(x + 0.16, oracle, width=0.32, color="#B9BDC1", label="Candidate oracle")
    ax_f.set_xticks(x, [str(level) for level in levels])
    ax_f.set_xlabel("Validation level")
    ax_f.set_ylabel("Held-out nRMSE")
    ax_f.set_title("Oracle ceiling diagnostic", fontsize=8, loc="left")
    ax_f.legend(fontsize=5.8)
    ax_f.grid(axis="y", color="#E9E9E9", lw=0.45)

    for label, axis in zip("abcdef", axes.ravel()):
        panel_label(axis, label)
    fig.suptitle("F-16 GVT 7.3 Hz nonlinear-modal pilot", y=0.995, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.91, wspace=0.34, hspace=0.42)
    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(args.output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    fields = ["panel", "series", "x", "value"]
    with args.source_data.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    manifest = {
        "figure": args.output_base.name,
        "backend": "Python/matplotlib",
        "pilot_admitted": admission["pilot_admitted"],
        "run_count": admission["run_count"],
        "source_data": args.source_data.name,
        "manuscript_modified": False,
    }
    args.output_base.with_name(args.output_base.name + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
