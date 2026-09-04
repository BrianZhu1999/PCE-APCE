"""Create an audit-only comparison figure from the admission summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "old_temporal": "Peak-height temporal",
    "old_state": "Peak-height state",
    "joint_quality": "Joint-pair quality",
    "joint_state": "Joint-pair state",
    "samepair_quality": "Same-pair quality",
    "samepair_state": "Same-pair state",
}


COLORS = {
    "old_temporal": "#3B6FB6",
    "old_state": "#2A9D8F",
    "joint_quality": "#D55E00",
    "joint_state": "#A23B72",
    "samepair_quality": "#E9C46A",
    "samepair_state": "#6C757D",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    experiments = payload["experiments"]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), constrained_layout=True)
    ax = axes[0]
    for key in ("old_temporal", "joint_quality", "samepair_quality"):
        rows = experiments[key]["per_frame"]
        ax.plot(
            [row["frame"] for row in rows],
            [row["mean_error_m"] for row in rows],
            color=COLORS[key],
            linewidth=1.8,
            label=LABELS[key],
        )
    ax.axhline(150.0, color="black", linestyle="--", linewidth=1.2, label="150 m admission gate")
    ax.set_ylabel("Mean 3-D position error (m)")
    ax.set_xlabel("Acoustic update frame")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.legend(ncol=2, frameon=True, fontsize=9)
    ax.set_title("a  Acoustic path error by frame", loc="left", fontsize=12, fontweight="bold")

    ax = axes[1]
    keys = ["old_temporal", "old_state", "joint_quality", "joint_state", "samepair_quality", "samepair_state"]
    means = [float(experiments[key]["reported_summary"]["offline_gps_mean_error_m"]) for key in keys]
    p90 = [float(experiments[key]["reported_summary"]["offline_gps_p90_error_m"]) for key in keys]
    x = np.arange(len(keys))
    width = 0.36
    ax.bar(x - width / 2, means, width, color=[COLORS[key] for key in keys], edgecolor="black", linewidth=0.5, label="Mean")
    ax.bar(x + width / 2, p90, width, color="white", edgecolor=[COLORS[key] for key in keys], linewidth=1.5, label="P90")
    ax.axhline(150.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, [LABELS[key].replace(" ", "\n", 1) for key in keys])
    ax.set_ylabel("3-D position error (m)")
    ax.set_ylim(0, max(p90) * 1.12)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=True, ncol=2)
    ax.set_title("b  Frozen-protocol admission comparison", loc="left", fontsize=12, fontweight="bold")
    fig.suptitle("Baoding three-source front-end audit (offline GPS scoring only)", fontsize=13, fontweight="bold")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
