from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES = ("wave", "spring", "heat")
FAMILIES = (("pce_shadow", "pce_analysis", "PCE"), ("apce_shadow", "apce_analysis", "APCE"))
COLORS = {"PCE": "#4C78A8", "APCE": "#F28E2B"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def trace_path(root: Path, case: str, method: str, seed: int) -> Path:
    return root / "artifacts" / "method_traces" / case / method / f"seed_{seed}.npz"


def median_trace(root: Path, case: str, method: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    traces = []
    for seed in range(2026080700, 2026080705):
        path = trace_path(root, case, method, seed)
        with np.load(path, allow_pickle=False) as data:
            traces.append((np.asarray(data[f"mechanism_local_progress"]), np.asarray(data[f"mechanism_local_{key}"])))
    progress = traces[0][0]
    values = np.stack([item[1] for item in traces], axis=0)
    return progress, np.median(values, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.root / "aggregate" / "figure2_corrected_formal_run_source_data.csv")
    grouped = {(row["case"], row["method"], int(row["seed"])): row for row in rows}

    fig, axes = plt.subplots(3, 2, figsize=(12.6, 9.0), dpi=600, constrained_layout=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42})
    for row_index, case in enumerate(CASES):
        ax = axes[row_index, 0]
        for shadow, _, label in FAMILIES:
            progress, ratio = median_trace(args.root, case, shadow, "erasure_ratio")
            ax.plot(progress, ratio, lw=1.35, color=COLORS[label], label=label)
        ax.axhline(1.0, color="#202020", lw=0.8, ls="--")
        ax.set_title(case.capitalize(), fontsize=12)
        ax.set_ylabel("Analysis / shadow separation", fontsize=10)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", color="#D9DDE2", lw=0.45, alpha=0.7)
        if row_index == 2:
            ax.set_xlabel("Normalized assimilation time", fontsize=10)
        if row_index == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")

        ax = axes[row_index, 1]
        x = np.arange(2)
        width = 0.36
        for j, (shadow, analysis, label) in enumerate(FAMILIES):
            dn = np.asarray([float(grouped[(case, analysis, seed)]["nrmse"]) - float(grouped[(case, shadow, seed)]["nrmse"]) for seed in range(2026080700, 2026080705)])
            dc = np.asarray([float(grouped[(case, analysis, seed)]["crps"]) - float(grouped[(case, shadow, seed)]["crps"]) for seed in range(2026080700, 2026080705)])
            ax.bar(x[0] + (j - 0.5) * width, dn.mean(), width, color=COLORS[label], label=label if row_index == 0 else None)
            ax.bar(x[1] + (j - 0.5) * width, dc.mean(), width, color=COLORS[label])
        ax.axhline(0.0, color="#202020", lw=0.8)
        ax.set_xticks(x, ["nRMSE", "CRPS"])
        ax.set_title(case.capitalize(), fontsize=12)
        ax.set_ylabel("Analysis - shadow", fontsize=10)
        ax.grid(axis="y", color="#D9DDE2", lw=0.45, alpha=0.7)
        if row_index == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")

    stem = args.output / "figure2_shadow_ablation_diagnostic"
    outputs = {".png": stem.with_suffix(".png"), ".pdf": stem.with_suffix(".pdf"), ".svg": stem.with_suffix(".svg"), ".tiff": stem.with_suffix(".tiff")}
    fig.savefig(outputs[".png"], dpi=600, facecolor="white")
    fig.savefig(outputs[".pdf"], facecolor="white")
    fig.savefig(outputs[".svg"], facecolor="white")
    fig.savefig(outputs[".tiff"], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    metadata = {"figure": "figure2_shadow_ablation_diagnostic", "scope": "5-seed smoke only", "source_root": str(args.root), "outputs": {k: str(v) for k, v in outputs.items()}, "output_sha256": {k: sha256(v) for k, v in outputs.items()}}
    (args.output / "figure2_shadow_ablation_diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
