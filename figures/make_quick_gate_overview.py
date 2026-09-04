from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "quick_gate_overview"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_LABEL = {
    "misspecified_forecast": "Wrong forecast",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "oracle_alpha": "Oracle-alpha",
    "pce": "PCE",
    "apce": "APCE",
}

METHOD_COLORS = {
    "misspecified_forecast": "#8F949B",
    "denkf": "#4C78A8",
    "letkf": "#7B6BB7",
    "oracle_alpha": "#54A24B",
    "pce": "#F2A541",
    "apce": "#D95F5F",
}

ORDER_FULL = [
    "misspecified_forecast",
    "denkf",
    "letkf",
    "oracle_alpha",
    "pce",
    "apce",
]
ORDER_WAVE = ["denkf", "letkf", "pce", "apce"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "None", "nan", "NaN"} else np.nan


def load_wave_runs() -> list[dict[str, float | str]]:
    rows = _read_csv(ROOT / "results_wave_repair_5seeds" / "runs.csv")
    allowed = set(ORDER_WAVE)
    out: list[dict[str, float | str]] = []
    for row in rows:
        method = row["method"]
        if method not in allowed:
            continue
        out.append(
            {
                "case": "wave",
                "method": method,
                "seed": row["seed"],
                "nrmse": _to_float(row, "displacement_nrmse"),
                "rmse": _to_float(row, "displacement_rmse"),
                "crps": _to_float(row, "crps"),
                "coverage_90": _to_float(row, "coverage_90"),
                "interval_width_90": _to_float(row, "interval_width_90"),
            }
        )
    return out


def load_spring_heat_runs() -> list[dict[str, float | str]]:
    rows = _read_csv(ROOT / "results_spring_heat_gate_5seeds" / "run_metrics.csv")
    allowed = set(ORDER_FULL)
    out: list[dict[str, float | str]] = []
    for row in rows:
        method = row["method"]
        if method not in allowed:
            continue
        out.append(
            {
                "case": row["case"],
                "method": method,
                "seed": row["seed"],
                "nrmse": _to_float(row, "nrmse"),
                "rmse": _to_float(row, "rmse"),
                "crps": _to_float(row, "crps"),
                "coverage_90": _to_float(row, "coverage_90"),
                "interval_width_90": _to_float(row, "interval_width_90"),
            }
        )
    return out


def mean_and_ci(values: list[float], *, rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        value = float(arr[0])
        return value, value, value
    samples = rng.choice(arr, size=(10000, arr.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(arr.mean()), float(low), float(high)


def summarize(rows: list[dict[str, float | str]], case: str, order: list[str]) -> dict[str, dict[str, tuple[float, float, float] | list[float]]]:
    rng = np.random.default_rng(20260806)
    case_rows = [r for r in rows if r["case"] == case]
    summary: dict[str, dict[str, tuple[float, float, float] | list[float]]] = {}
    for method in order:
        method_rows = [r for r in case_rows if r["method"] == method]
        if not method_rows:
            continue
        summary[method] = {
            "nrmse": mean_and_ci([float(r["nrmse"]) for r in method_rows], rng=rng),
            "crps": mean_and_ci([float(r["crps"]) for r in method_rows], rng=rng),
            "coverage_90": mean_and_ci([float(r["coverage_90"]) for r in method_rows], rng=rng),
            "interval_width_90": mean_and_ci(
                [float(r["interval_width_90"]) for r in method_rows], rng=rng
            ),
            "nrmse_values": [float(r["nrmse"]) for r in method_rows],
            "crps_values": [float(r["crps"]) for r in method_rows],
            "coverage_values": [float(r["coverage_90"]) for r in method_rows],
            "width_values": [float(r["interval_width_90"]) for r in method_rows],
        }
    return summary


def _set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def _draw_metric_bars(
    ax: plt.Axes,
    summary: dict[str, dict[str, tuple[float, float, float] | list[float]]],
    methods: list[str],
    metric: str,
    ylabel: str,
    *,
    scale: float = 1.0,
    title: str | None = None,
) -> None:
    x = np.arange(len(methods))
    means = np.asarray([summary[m][metric][0] for m in methods], dtype=float) * scale  # type: ignore[index]
    lows = np.asarray([summary[m][metric][1] for m in methods], dtype=float) * scale  # type: ignore[index]
    highs = np.asarray([summary[m][metric][2] for m in methods], dtype=float) * scale  # type: ignore[index]
    colors = [METHOD_COLORS[m] for m in methods]
    ax.bar(x, means, color=colors, width=0.64, edgecolor="white", linewidth=0.8, zorder=2)
    yerr = np.vstack([means - lows, highs - means])
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="none",
        ecolor="#2B2B2B",
        elinewidth=1.4,
        capsize=4.5,
        capthick=1.4,
        zorder=3,
    )
    jitter = np.linspace(-0.16, 0.16, 5)
    for idx, method in enumerate(methods):
        key = f"{metric}_values" if metric != "coverage_90" else "coverage_values"
        if metric == "interval_width_90":
            key = "width_values"
        values = np.asarray(summary[method][key], dtype=float) * scale  # type: ignore[index]
        ax.scatter(
            idx + jitter[: values.size],
            values,
            s=18,
            facecolor="white",
            edgecolor="#2B2B2B",
            linewidth=0.6,
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=6)
    ax.grid(axis="y", color="#D8DCE2", linewidth=0.6, alpha=0.8, zorder=0)


def plot_case(case: str, rows: list[dict[str, float | str]], order: list[str]) -> dict[str, str]:
    summary = summarize(rows, case, order)
    methods = [m for m in order if m in summary]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.4, 2.7),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08], "wspace": 0.42},
    )
    fig.patch.set_facecolor("white")

    title_case = {"wave": "Wave", "spring": "Spring oscillator", "heat": "Heat conduction"}[case]
    _draw_metric_bars(
        axes[0],
        summary,
        methods,
        "nrmse",
        "nRMSE (%)",
        scale=100.0,
        title=f"{title_case}: state error",
    )
    _draw_metric_bars(axes[1], summary, methods, "crps", "CRPS", title="Probabilistic loss")
    _draw_metric_bars(
        axes[2],
        summary,
        methods,
        "coverage_90",
        "90% coverage (%)",
        scale=100.0,
        title="Uncertainty coverage",
    )
    axes[2].axhline(90.0, color="#222222", linestyle=(0, (3, 2)), linewidth=1.0)
    axes[2].text(
        0.02,
        0.92,
        "Nominal 90%",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#222222",
    )

    for letter, ax in zip(["a", "b", "c"], axes):
        ax.text(
            -0.18,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="left",
        )

    fig.suptitle(f"{title_case} 5-seed gate quicklook", x=0.01, y=1.05, ha="left", fontsize=10.5)
    base = OUT / f"quicklook_{case}_gate"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    source_rows = []
    for method in methods:
        item = summary[method]
        source_rows.append(
            {
                "case": case,
                "method": method,
                "label": METHOD_LABEL[method],
                "nrmse_mean": item["nrmse"][0],  # type: ignore[index]
                "nrmse_ci_low": item["nrmse"][1],  # type: ignore[index]
                "nrmse_ci_high": item["nrmse"][2],  # type: ignore[index]
                "crps_mean": item["crps"][0],  # type: ignore[index]
                "crps_ci_low": item["crps"][1],  # type: ignore[index]
                "crps_ci_high": item["crps"][2],  # type: ignore[index]
                "coverage_90_mean": item["coverage_90"][0],  # type: ignore[index]
                "coverage_90_ci_low": item["coverage_90"][1],  # type: ignore[index]
                "coverage_90_ci_high": item["coverage_90"][2],  # type: ignore[index]
                "interval_width_90_mean": item["interval_width_90"][0],  # type: ignore[index]
            }
        )
    with (OUT / f"quicklook_{case}_gate_source.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    return {
        "svg": str(base.with_suffix(".svg")),
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "tiff": str(base.with_suffix(".tiff")),
    }


def plot_combined(rows: list[dict[str, float | str]]) -> dict[str, str]:
    cases = ["wave", "spring", "heat"]
    orders = {"wave": ORDER_WAVE, "spring": ORDER_FULL, "heat": ORDER_FULL}
    labels = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.2), gridspec_kw={"hspace": 0.72, "wspace": 0.38})
    for r, case in enumerate(cases):
        summary = summarize(rows, case, orders[case])
        methods = [m for m in orders[case] if m in summary]
        _draw_metric_bars(axes[r, 0], summary, methods, "nrmse", "nRMSE (%)", scale=100.0)
        _draw_metric_bars(axes[r, 1], summary, methods, "crps", "CRPS")
        _draw_metric_bars(axes[r, 2], summary, methods, "coverage_90", "Coverage (%)", scale=100.0)
        axes[r, 2].axhline(90.0, color="#222222", linestyle=(0, (3, 2)), linewidth=1.0)
        axes[r, 0].text(
            -0.32,
            1.05,
            labels[case],
            transform=axes[r, 0].transAxes,
            fontsize=10,
            fontweight="normal",
            ha="left",
            va="top",
        )
    for c, title in enumerate(["State error", "Probabilistic loss", "Coverage"]):
        axes[0, c].set_title(title, pad=8)
    base = OUT / "quicklook_wave_spring_heat_gate"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {
        "svg": str(base.with_suffix(".svg")),
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
        "tiff": str(base.with_suffix(".tiff")),
    }


def main() -> None:
    _set_pub_style()
    rows = load_wave_runs() + load_spring_heat_runs()
    outputs = {
        "wave": plot_case("wave", rows, ORDER_WAVE),
        "spring": plot_case("spring", rows, ORDER_FULL),
        "heat": plot_case("heat", rows, ORDER_FULL),
        "combined": plot_combined(rows),
    }
    with (OUT / "quick_gate_overview_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, ensure_ascii=False)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
