from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected5_freq1to8_screen_5seeds_20260813"
)
OUT = ROOT / "combined_analysis"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical",
    "pendulum": "Pendulum",
    "fhn": "FHN",
    "robertson": "Robertson",
}
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}


def sem(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def load_all() -> pd.DataFrame:
    frames = []
    for i in range(1, 9):
        p = ROOT / f"freq{i}" / "source_data" / "figure3_freq_sweep_run_source_data.csv"
        if not p.is_file():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df["observation_scenario"] = f"freq{i}"
        df["obs_level"] = i
        frames.append(df)
    all_runs = pd.concat(frames, ignore_index=True)
    all_runs = all_runs[
        all_runs["case"].isin(CASES)
        & all_runs["method"].isin(["aug_enkf", "bma_static", "pce", "apce"])
    ].copy()
    all_runs["case_label"] = all_runs["case"].map(CASE_LABELS)
    all_runs["method_label"] = all_runs["method"].map(METHOD_LABELS)
    return all_runs


def summarize(all_runs: pd.DataFrame):
    method_summary = (
        all_runs.groupby(["obs_level", "observation_scenario", "case", "method"], as_index=False)
        .agg(
            nrmse_mean=("nrmse", "mean"),
            nrmse_sem=("nrmse", sem),
            crps_mean=("crps", "mean"),
            crps_sem=("crps", sem),
            alpha_mae_mean=("alpha_absolute_error", "mean"),
            alpha_mae_sem=("alpha_absolute_error", sem),
            coverage_mean=("coverage_90", "mean"),
            width_mean=("interval_width_90", "mean"),
            runtime_mean=("core_runtime_seconds", "mean"),
            valid=("numerical_status", lambda x: int((x == "valid").sum())),
        )
        .sort_values(["obs_level", "case", "method"])
    )

    rows = []
    for (obs_level, scenario, case), group in method_summary.groupby(
        ["obs_level", "observation_scenario", "case"], sort=False
    ):
        pce_apce = group[group["method"].isin(["pce", "apce"])]
        best_n = pce_apce.loc[pce_apce["nrmse_mean"].idxmin()]
        best_a = pce_apce.loc[pce_apce["alpha_mae_mean"].idxmin()]
        aug = group[group["method"] == "aug_enkf"].iloc[0]
        bma = group[group["method"] == "bma_static"].iloc[0]
        rows.append(
            {
                "obs_level": int(obs_level),
                "observation_scenario": scenario,
                "case": case,
                "case_label": CASE_LABELS[case],
                "best_nrmse_method": METHOD_LABELS[str(best_n["method"])],
                "best_nrmse": float(best_n["nrmse_mean"]),
                "aug_nrmse": float(aug["nrmse_mean"]),
                "bma_nrmse": float(bma["nrmse_mean"]),
                "delta_nrmse_vs_aug": float(best_n["nrmse_mean"] - aug["nrmse_mean"]),
                "delta_nrmse_vs_bma": float(best_n["nrmse_mean"] - bma["nrmse_mean"]),
                "best_alpha_method": METHOD_LABELS[str(best_a["method"])],
                "best_alpha_mae": float(best_a["alpha_mae_mean"]),
                "aug_alpha_mae": float(aug["alpha_mae_mean"]),
                "bma_alpha_mae": float(bma["alpha_mae_mean"]),
                "delta_alpha_vs_aug": float(best_a["alpha_mae_mean"] - aug["alpha_mae_mean"]),
                "delta_alpha_vs_bma": float(best_a["alpha_mae_mean"] - bma["alpha_mae_mean"]),
            }
        )
    delta = pd.DataFrame(rows).sort_values(["obs_level", "case"])

    win_rows = []
    for obs_level, group in delta.groupby("obs_level", sort=True):
        win_rows.append(
            {
                "obs_level": int(obs_level),
                "wins_nrmse_vs_aug": int((group["delta_nrmse_vs_aug"] < 0).sum()),
                "wins_nrmse_vs_bma": int((group["delta_nrmse_vs_bma"] < 0).sum()),
                "wins_alpha_vs_aug": int((group["delta_alpha_vs_aug"] < 0).sum()),
                "wins_alpha_vs_bma": int((group["delta_alpha_vs_bma"] < 0).sum()),
                "mean_delta_nrmse_vs_aug": float(group["delta_nrmse_vs_aug"].mean()),
                "mean_delta_nrmse_vs_bma": float(group["delta_nrmse_vs_bma"].mean()),
                "mean_delta_alpha_vs_aug": float(group["delta_alpha_vs_aug"].mean()),
                "mean_delta_alpha_vs_bma": float(group["delta_alpha_vs_bma"].mean()),
            }
        )
    wins = pd.DataFrame(win_rows)
    return method_summary, delta, wins


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3, pad=2)
    ax.grid(False)


def make_plot(delta: pd.DataFrame, wins: pd.DataFrame):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    case_palette = {
        "pk_infusion": "#4477AA",
        "chemical": "#66A61E",
        "pendulum": "#AA3377",
        "fhn": "#CCBB44",
        "robertson": "#228833",
    }
    fig = plt.figure(figsize=(9.6, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.35, 0.9], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    x = np.arange(1, 9)
    for case in CASES:
        sub = delta[delta["case"] == case].sort_values("obs_level")
        ax0.plot(
            sub["obs_level"],
            sub["delta_nrmse_vs_aug"],
            lw=1.8,
            marker="o",
            ms=3.8,
            color=case_palette[case],
            label=CASE_LABELS[case],
        )
    ax0.axhline(0, color="#2A2A2A", lw=0.8)
    ymin = min(-0.035, float(delta["delta_nrmse_vs_aug"].min()) * 1.18)
    ymax = max(0.006, float(delta["delta_nrmse_vs_aug"].max()) * 1.18)
    ax0.set_ylim(ymin, ymax)
    ax0.fill_between([0.8, 8.2], [ymin, ymin], [0, 0], color="#E66100", alpha=0.08, lw=0)
    ax0.set_xlim(0.75, 8.25)
    ax0.set_xticks(x)
    ax0.set_xlabel("Observation interval multiplier")
    ax0.set_ylabel("Δ nRMSE vs Aug-EnKF")
    ax0.set_title("State reconstruction")
    ax0.text(0.88, ymin + 0.05 * (ymax - ymin), "PCE/APCE better", color="#E66100", fontsize=8)
    style_ax(ax0)
    ax0.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.03), handlelength=1.4)

    for case in CASES:
        sub = delta[delta["case"] == case].sort_values("obs_level")
        ax1.plot(
            sub["obs_level"],
            sub["delta_alpha_vs_aug"],
            lw=1.8,
            marker="o",
            ms=3.8,
            color=case_palette[case],
        )
    ax1.axhline(0, color="#2A2A2A", lw=0.8)
    ymin_a = min(-0.33, float(delta["delta_alpha_vs_aug"].min()) * 1.12)
    ymax_a = max(0.06, float(delta["delta_alpha_vs_aug"].max()) * 1.16)
    ax1.set_ylim(ymin_a, ymax_a)
    ax1.fill_between([0.8, 8.2], [ymin_a, ymin_a], [0, 0], color="#E66100", alpha=0.08, lw=0)
    ax1.set_xlim(0.75, 8.25)
    ax1.set_xticks(x)
    ax1.set_xlabel("Observation interval multiplier")
    ax1.set_ylabel("Δ alpha MAE vs Aug-EnKF")
    ax1.set_title("Cognitive-coordinate identification")
    style_ax(ax1)

    ax2.plot(
        wins["obs_level"],
        wins["wins_nrmse_vs_aug"],
        color="#E66100",
        marker="o",
        ms=4.0,
        lw=2.0,
        label="nRMSE",
    )
    ax2.plot(
        wins["obs_level"],
        wins["wins_alpha_vs_aug"],
        color="#CC79A7",
        marker="s",
        ms=3.8,
        lw=1.8,
        label="alpha MAE",
    )
    ax2.set_xlim(0.75, 8.25)
    ax2.set_ylim(-0.2, 5.35)
    ax2.set_xticks(x)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.set_xlabel("Observation interval multiplier")
    ax2.set_ylabel("Wins over Aug-EnKF / 5")
    ax2.set_title("Case-wise wins")
    style_ax(ax2)
    ax2.legend(frameon=False, loc="lower right", handlelength=1.4)

    for ax, label in zip([ax0, ax1, ax2], ["a", "b", "c"]):
        ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    fig.suptitle(
        "PCE/APCE become more competitive as observations are thinned (5 paired seeds)",
        x=0.52,
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.83)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = load_all()
    method_summary, delta, wins = summarize(all_runs)
    all_runs.to_csv(OUT / "figure3_freq1to8_all_runs_5seed.csv", index=False)
    method_summary.to_csv(OUT / "figure3_freq1to8_method_summary_5seed.csv", index=False)
    delta.to_csv(OUT / "figure3_freq1to8_delta_summary_5seed.csv", index=False)
    wins.to_csv(OUT / "figure3_freq1to8_win_summary_5seed.csv", index=False)

    manifest_paths = sorted(ROOT.glob("freq*/figure3_freq_sweep_manifest.json"))
    manifest = {
        "protocol": "figure3-selected5-freq1to8-combined-preview-5seed",
        "root": str(ROOT),
        "scenario_count": 8,
        "run_count": int(len(all_runs)),
        "valid_count": int((all_runs["numerical_status"] == "valid").sum()),
        "manifests": [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths],
    }
    (OUT / "figure3_freq1to8_combined_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig = make_plot(delta, wins)
    stem = OUT / "figure3_freq1to8_trend_preview_5seed"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("OUT", OUT)
    print("RUNS", len(all_runs), "VALID", int((all_runs["numerical_status"] == "valid").sum()))
    print("WINS")
    print(wins.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("DELTA_HEAD")
    print(
        delta[
            [
                "obs_level",
                "case",
                "best_nrmse_method",
                "delta_nrmse_vs_aug",
                "delta_nrmse_vs_bma",
                "best_alpha_method",
                "delta_alpha_vs_aug",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
