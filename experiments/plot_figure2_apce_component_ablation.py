from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["pce_baseline", "apce_full", "apce_no_dim", "apce_fixed_temp", "apce_no_forgetting", "apce_no_entropy"]
LABELS = {
    "pce_baseline": "PCE",
    "apce_full": "APCE",
    "apce_no_dim": "no dimension weighting",
    "apce_fixed_temp": "fixed temperature",
    "apce_no_forgetting": "no forgetting",
    "apce_no_entropy": "no entropy floor",
}
COLORS = ["#4C78A8", "#F28E2B", "#8A9A5B", "#B07AA1", "#6B7280", "#D07C38"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    frame["label"] = frame["method"].map(LABELS)
    metrics = [
        ("nrmse_mean", "nRMSE", True),
        ("crps_mean", "CRPS", True),
        ("coverage_error_90_mean", "coverage error", True),
        ("interval_width_90_mean", "interval width", True),
        ("alpha_absolute_error_mean", "alpha error", True),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(15.5, 8.2), dpi=600, constrained_layout=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    for row, case in enumerate(["wave", "spring", "heat"]):
        subset = frame[frame["case"] == case].set_index("method").reindex(METHODS)
        for col, (key, title, lower) in enumerate(metrics):
            ax = axes[row, col]
            vals = subset[key].to_numpy(dtype=float)
            ax.bar(np.arange(len(METHODS)), vals, color=COLORS, width=0.78)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(np.arange(len(METHODS)), ["PCE", "APCE", "-dim", "-temp", "-forget", "-entropy"], rotation=45, ha="right", fontsize=7)
            ax.grid(axis="y", color="#D9DDE2", lw=0.4, alpha=0.7)
            ax.set_ylabel(case.capitalize() if col == 0 else "")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 2 APCE component ablation: 5-seed smoke", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
