from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FULL_CSV = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected5_observation_trend_full_5seeds_20260813/"
    "source_data/figure3_obs_full5_run_source_data.csv"
)
SCREEN_CSV = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected6_observation_insufficient_screen_5seeds_20260813/"
    "combined_analysis/figure3_obs_insufficient_all_runs.csv"
)
OUT_DIR = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected5_observation_trend_preview_5seeds_20260813"
)

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical",
    "pendulum": "Pendulum",
    "fhn": "FHN",
    "robertson": "Robertson",
}
METHODS = ["aug_enkf", "bma_static", "pce", "apce"]
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
SCENARIOS = ["full", "freq2", "freq4"]
SCENARIO_LABELS = {"full": "1×", "freq2": "2×", "freq4": "4×"}


def sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def best_new(row_group: pd.DataFrame, metric: str) -> pd.Series:
    candidates = row_group[row_group["method"].isin(["pce", "apce"])]
    idx = candidates[metric].astype(float).idxmin()
    return candidates.loc[idx]


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(FULL_CSV)
    full = full[full["case"].isin(CASES) & full["method"].isin(METHODS)].copy()
    full["observation_scenario"] = "full"
    full["obs_interval_factor"] = 1

    screen = pd.read_csv(SCREEN_CSV)
    screen = screen[
        screen["case"].isin(CASES)
        & screen["method"].isin(METHODS)
        & screen["observation_scenario"].isin(["freq2", "freq4"])
    ].copy()

    all_runs = pd.concat([full, screen], ignore_index=True)
    all_runs["method_label"] = all_runs["method"].map(METHOD_LABELS)
    all_runs["case_label"] = all_runs["case"].map(CASE_LABELS)
    all_runs["obs_level"] = all_runs["observation_scenario"].map(
        {"full": 1, "freq2": 2, "freq4": 4}
    )

    method_summary = (
        all_runs.groupby(["observation_scenario", "obs_level", "case", "method"], as_index=False)
        .agg(
            nrmse_mean=("nrmse", "mean"),
            nrmse_sem=("nrmse", sem),
            alpha_mae_mean=("alpha_absolute_error", "mean"),
            alpha_mae_sem=("alpha_absolute_error", sem),
            crps_mean=("crps", "mean"),
            runtime_mean=("core_runtime_seconds", "mean"),
            valid=("numerical_status", lambda x: int((x == "valid").sum())),
        )
        .sort_values(["obs_level", "case", "method"])
    )

    rows = []
    for (scenario, obs_level, case), group in method_summary.groupby(
        ["observation_scenario", "obs_level", "case"], sort=False
    ):
        best = best_new(group, "nrmse_mean")
        aug = group[group["method"] == "aug_enkf"].iloc[0]
        bma = group[group["method"] == "bma_static"].iloc[0]
        rows.append(
            {
                "observation_scenario": scenario,
                "obs_level": int(obs_level),
                "case": case,
                "case_label": CASE_LABELS[case],
                "best_new_method": METHOD_LABELS[str(best["method"])],
                "best_new_nrmse": float(best["nrmse_mean"]),
                "aug_nrmse": float(aug["nrmse_mean"]),
                "bma_nrmse": float(bma["nrmse_mean"]),
                "delta_nrmse_vs_aug": float(best["nrmse_mean"] - aug["nrmse_mean"]),
                "delta_nrmse_vs_bma": float(best["nrmse_mean"] - bma["nrmse_mean"]),
                "best_new_alpha_mae": float(best["alpha_mae_mean"]),
                "aug_alpha_mae": float(aug["alpha_mae_mean"]),
                "bma_alpha_mae": float(bma["alpha_mae_mean"]),
                "delta_alpha_vs_aug": float(best["alpha_mae_mean"] - aug["alpha_mae_mean"]),
                "delta_alpha_vs_bma": float(best["alpha_mae_mean"] - bma["alpha_mae_mean"]),
            }
        )
    delta = pd.DataFrame(rows)

    win_rows = []
    for scenario, group in delta.groupby("observation_scenario"):
        obs_level = int(group["obs_level"].iloc[0])
        win_rows.append(
            {
                "observation_scenario": scenario,
                "obs_level": obs_level,
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
    wins = pd.DataFrame(win_rows).sort_values("obs_level")
    return all_runs, method_summary, delta.sort_values(["obs_level", "case"]), wins


def set_panel_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", width=0.8, length=3, pad=2)
    ax.grid(False)


def plot_preview(method_summary: pd.DataFrame, delta: pd.DataFrame, wins: pd.DataFrame) -> plt.Figure:
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
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.9,
        }
    )
    colors = {
        "best": "#E66100",
        "aug": "#4C78A8",
        "bma": "#5A5A5A",
        "pce": "#F28E2B",
        "apce": "#D1495B",
        "positive": "#4C78A8",
        "negative": "#D1495B",
    }

    fig = plt.figure(figsize=(9.2, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.2, 0.9], wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    x = np.array([1, 2, 4], dtype=float)
    xlabels = ["1×", "2×", "4×"]
    case_palette = {
        "pk_infusion": "#4477AA",
        "chemical": "#66A61E",
        "pendulum": "#AA3377",
        "fhn": "#CCBB44",
        "robertson": "#228833",
    }

    for case in CASES:
        sub = delta[delta["case"] == case].sort_values("obs_level")
        ax0.plot(
            sub["obs_level"],
            sub["delta_nrmse_vs_aug"],
            color=case_palette[case],
            lw=1.8,
            marker="o",
            ms=4.0,
            label=CASE_LABELS[case],
        )
    ax0.axhline(0, color="#2A2A2A", lw=0.8)
    ax0.fill_between([0.8, 4.2], [-0.03, -0.03], [0, 0], color="#E66100", alpha=0.08, lw=0)
    ax0.set_xlim(0.85, 4.15)
    ax0.set_xticks(x, xlabels)
    ax0.set_xlabel("Observation interval")
    ax0.set_ylabel("Δ nRMSE vs Aug-EnKF")
    ax0.set_title("State reconstruction advantage")
    ax0.text(0.88, -0.028, "PCE/APCE better", color=colors["best"], fontsize=8)
    set_panel_style(ax0)
    ax0.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=1, handlelength=1.4)

    for case in CASES:
        sub = delta[delta["case"] == case].sort_values("obs_level")
        ax1.plot(
            sub["obs_level"],
            sub["delta_alpha_vs_aug"],
            color=case_palette[case],
            lw=1.8,
            marker="o",
            ms=4.0,
        )
    ax1.axhline(0, color="#2A2A2A", lw=0.8)
    y_min = min(-0.02, float(delta["delta_alpha_vs_aug"].min()) * 1.15)
    y_max = max(0.02, float(delta["delta_alpha_vs_aug"].max()) * 1.15)
    ax1.fill_between([0.8, 4.2], [y_min, y_min], [0, 0], color="#E66100", alpha=0.08, lw=0)
    ax1.set_xlim(0.85, 4.15)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xticks(x, xlabels)
    ax1.set_xlabel("Observation interval")
    ax1.set_ylabel("Δ alpha MAE vs Aug-EnKF")
    ax1.set_title("Cognitive-parameter advantage")
    set_panel_style(ax1)

    wins = wins.sort_values("obs_level")
    xpos = np.arange(len(wins))
    width = 0.34
    ax2.bar(
        xpos - width / 2,
        wins["wins_nrmse_vs_aug"],
        width=width,
        color="#E66100",
        edgecolor="#8F3B00",
        linewidth=0.8,
        label="nRMSE",
    )
    ax2.bar(
        xpos + width / 2,
        wins["wins_alpha_vs_aug"],
        width=width,
        color="#CC79A7",
        edgecolor="#7A3F62",
        linewidth=0.8,
        label="alpha MAE",
    )
    ax2.set_xticks(xpos, [SCENARIO_LABELS[s] for s in wins["observation_scenario"]])
    ax2.set_ylim(0, 5.4)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.set_ylabel("Wins over Aug-EnKF / 5")
    ax2.set_title("Win count")
    set_panel_style(ax2)
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.02), handlelength=1.2)

    for ax, label in zip([ax0, ax1, ax2], ["a", "b", "c"]):
        ax.text(
            -0.16,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "PCE/APCE gains emerge as ODE observations become sparse (5 paired seeds)",
        x=0.51,
        y=1.03,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.82)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs, method_summary, delta, wins = build_tables()
    all_runs.to_csv(OUT_DIR / "figure3_obs_trend_all_runs_5seed.csv", index=False)
    method_summary.to_csv(OUT_DIR / "figure3_obs_trend_method_summary_5seed.csv", index=False)
    delta.to_csv(OUT_DIR / "figure3_obs_trend_delta_summary_5seed.csv", index=False)
    wins.to_csv(OUT_DIR / "figure3_obs_trend_win_summary_5seed.csv", index=False)

    fig = plot_preview(method_summary, delta, wins)
    stem = OUT_DIR / "figure3_obs_trend_preview_5seed"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("OUT_DIR", OUT_DIR)
    print("delta")
    print(
        delta[
            [
                "obs_level",
                "case",
                "best_new_method",
                "best_new_nrmse",
                "aug_nrmse",
                "delta_nrmse_vs_aug",
                "best_new_alpha_mae",
                "aug_alpha_mae",
                "delta_alpha_vs_aug",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("wins")
    print(wins.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
