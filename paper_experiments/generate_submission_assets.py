from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "ncs_chinese_submission"
FIGURES = SUBMISSION / "figures"
EXTENDED = SUBMISSION / "figures_extended"
SOURCE_DATA = SUBMISSION / "source_data"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def box(axis: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.13),
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=7)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=color))


def make_framework_figure() -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    blue, orange, green, grey = "#4C78A8", "#E09F3E", "#6AAE8B", "#7F8FA6"

    axis.text(0.02, 0.93, "a  双层随机--认知波场同化", fontsize=8, fontweight="bold")
    box(axis, (0.03, 0.62), 0.14, 0.17, "认知轨道\n$\\alpha_q\\mapsto\\theta_q$", orange)
    box(axis, (0.22, 0.62), 0.17, 0.17, "PDE 预测\n$\\mathcal{F}_{\\theta_q}$", blue)
    box(axis, (0.44, 0.62), 0.18, 0.17, "影子预测库\n不接受分析增量", orange)
    box(axis, (0.68, 0.62), 0.14, 0.17, "PCE/APCE\n预测证据", orange)
    box(axis, (0.86, 0.62), 0.11, 0.17, "轨道权重\n$\\omega_{q,k}$", orange)
    for x0, x1 in [(0.17, 0.22), (0.39, 0.44), (0.62, 0.68), (0.82, 0.86)]:
        arrow(axis, (x0, 0.705), (x1, 0.705))

    box(axis, (0.22, 0.25), 0.17, 0.17, "分析分支集合\n$\\{X_{q,k}^{(j)}\\}$", blue)
    box(axis, (0.44, 0.25), 0.18, 0.17, "观测子空间\nEnSF 反向扩散", green)
    box(axis, (0.68, 0.25), 0.14, 0.17, "LR 增量传播\n$C_{uo}(C_{oo}+\\lambda I)^{-1}$", green)
    box(axis, (0.86, 0.25), 0.11, 0.17, "混合波场\n$\\widehat X_k$", blue)
    for x0, x1 in [(0.39, 0.44), (0.62, 0.68), (0.82, 0.86)]:
        arrow(axis, (x0, 0.335), (x1, 0.335))
    arrow(axis, (0.305, 0.62), (0.305, 0.42))
    arrow(axis, (0.915, 0.62), (0.915, 0.42), orange)
    box(axis, (0.03, 0.25), 0.14, 0.17, "稀疏观测\n$y_k=H_kX_k+\\varepsilon_k$", grey)
    arrow(axis, (0.17, 0.335), (0.44, 0.335), grey)
    axis.text(0.03, 0.08, "外层：低维认知参数选择与校准", color=orange, fontweight="bold")
    axis.text(0.44, 0.08, "内层：高维随机状态分析", color=green, fontweight="bold")
    save_figure(figure, FIGURES / "figure1_framework")
    plt.close(figure)


def make_main_wave_figure() -> None:
    data = np.load(ROOT / "results_quick" / "simulation_data.npz")
    mc = load_csv(ROOT / "results_statistics_50seeds" / "monte_carlo_runs.csv")
    figure = plt.figure(figsize=(7.2, 5.1))
    grid = figure.add_gridspec(2, 3, height_ratios=[1.15, 1.0], wspace=0.38, hspace=0.40)
    truth = data["truth_field"]
    estimate = data["estimate_field"]
    error = np.abs(estimate - truth)
    extent = [float(data["x"][0]), float(data["x"][-1]), float(data["times"][0]), float(data["times"][-1])]
    vmax = float(np.max(np.abs(truth)))
    arrays = [(truth, "a  真值"), (estimate, "b  PCE 重建"), (error, "c  绝对误差")]
    for col, (array, title) in enumerate(arrays):
        axis = figure.add_subplot(grid[0, col])
        if col < 2:
            axis.imshow(array, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            axis.imshow(array, extent=extent, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("空间 $x$")
        if col == 0:
            axis.set_ylabel("时间 $t$")
        else:
            axis.set_yticklabels([])

    axis_rmse = figure.add_subplot(grid[1, 0])
    axis_rmse.plot(data["times"], data["rmse_baseline"], color="#9A9A9A", label="确定性基线")
    axis_rmse.plot(data["times"], data["rmse_hybrid"], color="#4C78A8", label="双层 EnSF-LR")
    axis_rmse.set(xlabel="时间", ylabel="RMSE")
    axis_rmse.set_title("d  单次误差演化", loc="left", fontweight="bold")
    axis_rmse.legend()

    axis_pair = figure.add_subplot(grid[1, 1])
    baseline = np.asarray([float(row["mean_rmse_baseline"]) for row in mc])
    hybrid = np.asarray([float(row["mean_rmse_hybrid"]) for row in mc])
    limit = max(float(np.max(baseline)), float(np.max(hybrid))) * 1.05
    axis_pair.scatter(baseline, hybrid, s=13, color="#4C78A8", alpha=0.75)
    axis_pair.plot([0, limit], [0, limit], linestyle="--", color="#777777", linewidth=0.8)
    axis_pair.set(xlabel="基线平均 RMSE", ylabel="双层平均 RMSE", xlim=(0, limit), ylim=(0, limit))
    axis_pair.set_title("e  50 种子配对结果", loc="left", fontweight="bold")

    axis_alpha = figure.add_subplot(grid[1, 2])
    alpha_values = sorted({float(row["alpha_best_final"]) for row in mc})
    counts = [sum(abs(float(row["alpha_best_final"]) - alpha) < 1.0e-12 for row in mc) for alpha in alpha_values]
    colors = ["#E09F3E" if abs(alpha - 0.78) < 1.0e-12 else "#B9C6D3" for alpha in alpha_values]
    axis_alpha.bar([f"{value:.2f}" for value in alpha_values], counts, color=colors)
    axis_alpha.set(xlabel=r"终止 $\alpha$ 轨道", ylabel="种子数")
    axis_alpha.set_title("f  轨道恢复", loc="left", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, FIGURES / "figure2_main_wave")
    plt.close(figure)


def make_ablation_figure() -> None:
    rows = load_csv(ROOT / "results_benchmark_v4_50seeds" / "v4_ablation_runs.csv")
    methods = ["A0_old", "A1_paired_init", "A2_paired_sampler", "A3_shadow", "A4_gaussian", "A5_shrinkage", "A6_pce", "A7_apce"]
    labels = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    colors = ["#9A9A9A", "#A8B3C1", "#A8B3C1", "#A8B3C1", "#C86B6B", "#6AAE8B", "#4C78A8", "#E09F3E"]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    data = [[float(row["mean_rmse"]) for row in rows if row["ablation"] == method] for method in methods]
    boxplot = axes[0, 0].boxplot(data, patch_artist=True, widths=0.65, showfliers=False)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0, 0].set_xticks(range(1, len(labels) + 1), labels)
    axes[0, 0].set(ylabel="平均 RMSE", title="a  递进组件消融")

    alpha_mae = [np.mean([float(row["alpha_abs_error"]) for row in rows if row["ablation"] == method]) for method in methods]
    axes[0, 1].bar(labels, alpha_mae, color=colors)
    axes[0, 1].set(ylabel=r"连续 $\alpha$ MAE", title="b  参数辨识")

    means = [np.mean(values) for values in data]
    finals = [np.mean([float(row["final_rmse"]) for row in rows if row["ablation"] == method]) for method in methods]
    for index, method in enumerate(methods):
        axes[1, 0].scatter(means[index], finals[index], color=colors[index], s=35)
        axes[1, 0].text(means[index], finals[index], labels[index], fontsize=7, ha="left", va="bottom")
    axes[1, 0].set(xlabel="平均 RMSE", ylabel="终止 RMSE", title="c  全时域--终止权衡")

    by_seed: dict[str, dict[str, float]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["ablation"]] = float(row["mean_rmse"])
    differences = np.asarray([values["A6_pce"] - values["A7_apce"] for values in by_seed.values()])
    axes[1, 1].hist(differences * 1.0e5, bins=12, color="#E09F3E", edgecolor="white")
    axes[1, 1].axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[1, 1].set(xlabel=r"$(\mathrm{RMSE}_{A6}-\mathrm{RMSE}_{A7})\times10^5$", ylabel="种子数", title="d  APCE 配对增益")
    for axis in axes.ravel():
        title = axis.get_title()
        axis.set_title("", loc="center")
        axis.set_title(title, loc="left", fontweight="bold")
    figure.tight_layout(w_pad=1.5, h_pad=1.5)
    save_figure(figure, FIGURES / "figure3_ablation")
    plt.close(figure)


def make_generalization_figure() -> None:
    full = load_json(ROOT / "results_paper_full_alpha" / "generalization_summary.json")
    completion = load_json(ROOT / "results_paper_completion" / "calibration" / "calibration_summary.json")
    broad = load_json(ROOT / "results_generalization_v4" / "generalization_summary.json")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    colors = {"A6_pce": "#4C78A8", "A7_apce": "#E09F3E"}
    labels = {"A6_pce": "PCE", "A7_apce": "APCE"}
    for method in ("A6_pce", "A7_apce"):
        subset = [row for row in full["summary"] if row["method"] == method]
        subset.sort(key=lambda row: row["alpha_true"])
        axes[0, 0].plot([row["alpha_true"] for row in subset], [row["mean_rmse"] for row in subset], marker="o", color=colors[method], label=labels[method])
        axes[0, 1].plot([row["alpha_true"] for row in subset], [row["alpha_mae"] for row in subset], marker="s", color=colors[method], label=labels[method])
    axes[0, 0].set(xlabel=r"真实 $\alpha$", ylabel="平均 RMSE", title="a  全轨道状态误差")
    axes[0, 1].set(xlabel=r"真实 $\alpha$", ylabel=r"连续 $\alpha$ MAE", title="b  全轨道参数误差")
    axes[0, 0].legend()

    levels = completion["levels"]
    axes[1, 0].plot(levels, levels, linestyle="--", color="#777777", linewidth=0.8, label="理想")
    for method in ("A6_pce", "A7_apce"):
        subset = [row for row in completion["summary"] if row["method"] == method]
        axes[1, 0].plot(levels, [row["picp"] for row in subset], marker="o", color=colors[method], label=labels[method])
    axes[1, 0].set(xlabel="名义覆盖率", ylabel="经验覆盖率", title="c  概率校准")

    alpha_reference = min(broad["alpha_values"], key=lambda value: abs(value - 0.78))
    noise_values = broad["obs_noise_values"]
    sensor_values = broad["sensor_values"]
    matrix = np.zeros((len(noise_values), len(sensor_values)))
    for i, noise in enumerate(noise_values):
        for j, sensors in enumerate(sensor_values):
            row = next(item for item in broad["summary"] if item["method"] == "A7_apce" and item["alpha_true"] == alpha_reference and item["obs_noise"] == noise and item["n_sensors"] == sensors)
            matrix[i, j] = row["mean_rmse"]
    image = axes[1, 1].imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    axes[1, 1].set_xticks(range(len(sensor_values)), [str(value) for value in sensor_values])
    axes[1, 1].set_yticks(range(len(noise_values)), [f"{value:.2f}" for value in noise_values])
    axes[1, 1].set(xlabel="传感器数", ylabel="观测噪声", title="d  稀疏与噪声应力测试")
    figure.colorbar(image, ax=axes[1, 1], label="APCE 平均 RMSE", fraction=0.046)
    for axis in axes.ravel():
        title = axis.get_title()
        axis.set_title("", loc="center")
        axis.set_title(title, loc="left", fontweight="bold")
    figure.tight_layout(w_pad=1.5, h_pad=1.5)
    save_figure(figure, FIGURES / "figure4_generalization")
    plt.close(figure)


def make_multiphysics_figure() -> None:
    summary = load_json(ROOT / "results_paper_multiphysics" / "multiphysics_summary.json")
    cases = ["acoustic2d", "burgers1d", "allen_cahn1d"]
    case_labels = ["二维声学", "Burgers", "Allen--Cahn"]
    methods = ["misspecified", "oracle", "pce", "apce"]
    method_labels = ["失配 EnSF-LR", r"Oracle-$\alpha$ EnSF-LR", "PCE", "APCE"]
    colors = ["#9A9A9A", "#6AAE8B", "#4C78A8", "#E09F3E"]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))

    x = np.arange(len(cases))
    width = 0.18
    for index, method in enumerate(methods):
        values = [next(row for row in summary["summary"] if row["case"] == case and row["method"] == method)["mean_rmse"] for case in cases]
        axes[0].bar(x + (index - 1.5) * width, values, width=width, color=colors[index], label=method_labels[index])
    axes[0].set_xticks(x, case_labels, rotation=22)
    axes[0].set_ylabel("时间平均 RMSE")
    axes[0].set_title("a  跨方程状态重建", loc="left", fontweight="bold")

    for method, color, label, marker in [("pce", colors[2], "PCE", "o"), ("apce", colors[3], "APCE", "s")]:
        values = [next(row for row in summary["summary"] if row["case"] == case and row["method"] == method)["alpha_mae"] for case in cases]
        axes[1].plot(x, values, marker=marker, color=color, label=label)
    axes[1].set_xticks(x, case_labels, rotation=22)
    axes[1].set_ylabel(r"连续 $\alpha$ MAE")
    axes[1].set_title("b  认知参数辨识", loc="left", fontweight="bold")

    for method, color, label, marker in [("pce", colors[2], "PCE", "o"), ("apce", colors[3], "APCE", "s")]:
        reductions = []
        for case in cases:
            baseline = next(row for row in summary["summary"] if row["case"] == case and row["method"] == "misspecified")["mean_rmse"]
            value = next(row for row in summary["summary"] if row["case"] == case and row["method"] == method)["mean_rmse"]
            reductions.append(100.0 * (1.0 - value / baseline))
        axes[2].plot(x, reductions, marker=marker, color=color, label=label)
    axes[2].axhline(0.0, color="#777777", linewidth=0.8)
    axes[2].set_xticks(x, case_labels, rotation=22)
    axes[2].set_ylabel("相对失配模型的 RMSE 降幅 (\%)")
    axes[2].set_title("c  认知校正收益", loc="left", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=1.4)
    save_figure(figure, FIGURES / "figure5_multiphysics")
    plt.close(figure)

    source_stem = ROOT / "results_paper_multiphysics" / "figure_multiphysics_fields"
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        shutil.copy2(source_stem.with_suffix(suffix), EXTENDED / f"extended_figure_multiphysics_fields{suffix}")


def make_completion_figure() -> None:
    mismatch = load_json(ROOT / "results_paper_completion" / "mismatch" / "mismatch_summary.json")
    tuning = load_json(ROOT / "results_paper_completion" / "fair_tuning" / "fair_tuning_summary.json")
    scaling = load_json(ROOT / "results_paper_completion" / "scalability" / "scalability_summary.json")
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    conditions = mismatch["conditions"]
    labels = ["匹配", "细/粗", "$c$", "阻尼", "源", "边界", "非高斯 Q", "非高斯 R"]
    colors = {"enkf": "#9A9A9A", "A6_pce": "#4C78A8", "A7_apce": "#E09F3E"}
    for method in ("enkf", "A6_pce", "A7_apce"):
        values = [next(row for row in mismatch["summary"] if row["mismatch"] == condition and row["method"] == method)["mean_rmse"] for condition in conditions]
        axes[0].plot(range(len(conditions)), values, marker="o", linewidth=1.0, color=colors[method], label={"enkf": "EnKF", "A6_pce": "PCE", "A7_apce": "APCE"}[method])
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(conditions)), labels, rotation=38, ha="right")
    axes[0].set(ylabel="平均 RMSE（对数）", title="a  模型失配")
    axes[0].legend()

    tune_labels = {"enkf": "EnKF", "ensf_direct": "EnSF", "ensf_lr": "EnSF-LR", "joint_param_enkf": "联合 EnKF", "pce": "PCE"}
    tune_colors = ["#9A9A9A", "#A8B3C1", "#7F8FA6", "#6AAE8B", "#4C78A8"]
    axes[1].bar([tune_labels[row["method"]] for row in tuning["summary"]], [row["mean_rmse"] for row in tuning["summary"]], color=tune_colors)
    axes[1].tick_params(axis="x", rotation=32)
    axes[1].set(ylabel="留出测试 RMSE", title="b  等预算调参")

    dimensions = [row for row in scaling["scaling"] if row["ensemble_size"] == 18 and row["n_alpha"] == 7 and row["reverse_steps"] == 8]
    axes[2].plot([row["state_dimension"] for row in dimensions], [row["runtime_seconds"] for row in dimensions], marker="s", color="#4C78A8", label="运行时间")
    twin = axes[2].twinx()
    twin.plot([row["state_dimension"] for row in dimensions], [row["peak_memory_mb"] for row in dimensions], marker="^", color="#E09F3E", label="峰值内存")
    axes[2].set(xlabel="状态维数", ylabel="运行时间 (s)", title="c  计算扩展性")
    twin.set_ylabel("峰值内存 (MB)")
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    for axis in axes:
        title = axis.get_title()
        axis.set_title("", loc="center")
        axis.set_title(title, loc="left", fontweight="bold")
    figure.tight_layout(w_pad=1.4)
    save_figure(figure, FIGURES / "figure6_robustness_cost")
    plt.close(figure)


def copy_extended_assets() -> None:
    mappings = {
        ROOT / "report_v4" / "figures" / "19_waveguide_final_pressure.png": EXTENDED / "extended_figure_waveguide_final.png",
        ROOT / "report_v4" / "figures" / "29_v4_entropy_rmse_tradeoff.png": EXTENDED / "extended_figure_entropy_tradeoff.png",
        ROOT / "report_v4" / "figures" / "31_v4_terminal_tradeoff.png": EXTENDED / "extended_figure_terminal_tradeoff.png",
    }
    for source, target in mappings.items():
        shutil.copy2(source, target)

    waveguide = np.load(ROOT / "results_acoustic_waveguide" / "waveguide_data.npz")
    x_pressure = waveguide["x_pressure"]
    times_ms = 1000.0 * waveguide["times"]
    truth = waveguide["truth_pressure"]
    estimate = waveguide["estimate_pressure"]
    arrays = [truth, estimate, np.abs(truth - estimate)]
    titles = ["a  压力真值", "b  双层估计", "c  绝对压力误差"]
    max_amplitude = max(np.abs(truth).max(), np.abs(estimate).max())
    figure = plt.figure(figsize=(9.0, 6.4))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]
    extent = [x_pressure[0], x_pressure[-1], times_ms[0], times_ms[-1]]
    for index, (axis, array, title) in enumerate(zip(axes, arrays, titles)):
        limits = {"vmin": -max_amplitude, "vmax": max_amplitude} if index < 2 else {}
        image = axis.imshow(array, origin="lower", aspect="auto", extent=extent, cmap="viridis", **limits)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("轴向位置 x (m)")
        axis.set_ylabel("时间 (ms)")
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    figure.tight_layout(h_pad=1.1, w_pad=1.3)
    save_figure(figure, EXTENDED / "extended_figure_waveguide_spacetime")
    plt.close(figure)


def copy_source_data() -> None:
    sources = {
        "monte_carlo_50seeds.csv": ROOT / "results_statistics_50seeds" / "monte_carlo_runs.csv",
        "baseline_benchmark_50seeds.csv": ROOT / "results_benchmark_v3_50seeds" / "benchmark_runs.csv",
        "v4_ablation_50seeds.csv": ROOT / "results_benchmark_v4_50seeds" / "v4_ablation_runs.csv",
        "noise_sensor_generalization.csv": ROOT / "results_generalization_v4" / "generalization_runs.csv",
        "full_alpha_generalization.csv": ROOT / "results_paper_full_alpha" / "generalization_runs.csv",
        "multiphysics_runs.csv": ROOT / "results_paper_multiphysics" / "multiphysics_runs.csv",
        "calibration_runs.csv": ROOT / "results_paper_completion" / "calibration" / "calibration_runs.csv",
        "mismatch_runs.csv": ROOT / "results_paper_completion" / "mismatch" / "mismatch_runs.csv",
        "fair_tuning_runs.csv": ROOT / "results_paper_completion" / "fair_tuning" / "fair_tuning_runs.csv",
        "scalability_runs.csv": ROOT / "results_paper_completion" / "scalability" / "scalability_runs.csv",
        "parallel_speedup.csv": ROOT / "results_paper_completion" / "scalability" / "parallel_speedup.csv",
        "waveguide_metrics.json": ROOT / "results_acoustic_waveguide" / "waveguide_metrics.json",
        "waveguide_data.npz": ROOT / "results_acoustic_waveguide" / "waveguide_data.npz",
    }
    for target_name, source in sources.items():
        shutil.copy2(source, SOURCE_DATA / target_name)


def format_percent(value: float) -> str:
    return f"{value:.1f}\\%"


def latex_scientific(value: float, digits: int = 4) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"\\ensuremath{{{mantissa}\\times10^{{{int(exponent)}}}}}"


def write_macros_and_tables() -> None:
    mc = load_json(ROOT / "results_statistics_50seeds" / "monte_carlo_summary.json")
    v4_summary = load_json(ROOT / "results_benchmark_v4_50seeds" / "v4_summary.json")
    full_alpha = load_json(ROOT / "results_paper_full_alpha" / "generalization_summary.json")
    multi = load_json(ROOT / "results_paper_multiphysics" / "multiphysics_summary.json")
    completion = load_json(ROOT / "results_paper_completion" / "completion_summary.json")
    waveguide = load_json(ROOT / "results_acoustic_waveguide" / "waveguide_metrics.json")
    waveguide_config = waveguide["config"]
    waveguide_metrics = waveguide["metrics"]
    pce = v4_summary["summaries"]["A6_pce"]
    apce = v4_summary["summaries"]["A7_apce"]
    calibration90 = [item for item in completion["calibration"]["summary"] if abs(item["nominal_coverage"] - 0.90) < 1.0e-12]
    multi_reductions = {}
    for case in multi["cases"]:
        baseline = next(item for item in multi["summary"] if item["case"] == case and item["method"] == "misspecified")
        method = next(item for item in multi["summary"] if item["case"] == case and item["method"] == "pce")
        multi_reductions[case] = 100.0 * (1.0 - method["mean_rmse"] / baseline["mean_rmse"])
    macros = [
        f"\\newcommand{{\\PaperMCSeeds}}{{{mc['n_seeds']}}}",
        f"\\newcommand{{\\PaperMCReduction}}{{{mc['relative_reduction_percent_mean']:.2f}\\%}}",
        f"\\newcommand{{\\PaperMCWinRate}}{{{mc['win_rate_percent']:.1f}\\%}}",
        f"\\newcommand{{\\PaperPCErmse}}{{{pce['mean_rmse_mean']:.4e}}}",
        f"\\newcommand{{\\PaperAPCErmse}}{{{apce['mean_rmse_mean']:.4e}}}",
        f"\\newcommand{{\\PaperPCETopOne}}{{{pce['alpha_top1_accuracy_percent']:.1f}\\%}}",
        f"\\newcommand{{\\PaperAPCETopOne}}{{{apce['alpha_top1_accuracy_percent']:.1f}\\%}}",
        f"\\newcommand{{\\PaperPCECoverageNinety}}{{{next(item for item in calibration90 if item['method']=='A6_pce')['picp']*100:.1f}\\%}}",
        f"\\newcommand{{\\PaperAPCECoverageNinety}}{{{next(item for item in calibration90 if item['method']=='A7_apce')['picp']*100:.1f}\\%}}",
        f"\\newcommand{{\\PaperAcousticReduction}}{{{multi_reductions['acoustic2d']:.1f}\\%}}",
        f"\\newcommand{{\\PaperBurgersReduction}}{{{multi_reductions['burgers1d']:.1f}\\%}}",
        f"\\newcommand{{\\PaperAllenReduction}}{{{multi_reductions['allen_cahn1d']:.1f}\\%}}",
        f"\\newcommand{{\\PaperFullAlphaConditions}}{{{len(full_alpha['alpha_values'])}}}",
        f"\\newcommand{{\\PaperWaveguideLength}}{{{waveguide_config['length']:.2f}}}",
        f"\\newcommand{{\\PaperWaveguideDiameter}}{{{waveguide_config['diameter']:.2f}}}",
        f"\\newcommand{{\\PaperWaveguideRMSE}}{{{latex_scientific(waveguide_metrics['mean_rmse_hybrid_pa'])}}}",
        f"\\newcommand{{\\PaperWaveguideBaselineRMSE}}{{{latex_scientific(waveguide_metrics['mean_rmse_baseline_pa'])}}}",
        f"\\newcommand{{\\PaperWaveguideReduction}}{{{waveguide_metrics['relative_rmse_reduction_percent']:.2f}\\%}}",
        f"\\newcommand{{\\PaperWaveguideCutoff}}{{{waveguide_metrics['plane_wave_cutoff_hz']:.1f}}}",
    ]
    (SUBMISSION / "results_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    case_labels = {
        "acoustic2d": "二维声学",
        "burgers1d": "Burgers",
        "allen_cahn1d": "Allen--Cahn",
    }
    method_labels = {
        "misspecified": "失配 EnSF-LR",
        "oracle": "Oracle-$\\alpha$ EnSF-LR",
        "pce": "PCE",
        "apce": "APCE",
    }
    rows = []
    for item in multi["summary"]:
        if item["method"] not in {"misspecified", "oracle", "pce", "apce"}:
            continue
        rows.append(
            f"{case_labels[item['case']]} & {method_labels[item['method']]} & {item['state_dimension']} & {item['mean_rmse']:.4e} & {item['final_rmse']:.4e} & {item['alpha_mae']:.4f} \\\\"
        )
    (SUBMISSION / "table_multiphysics.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8"
    )

    mismatch_labels = {
        "matched": "匹配",
        "fine_grid": "细--粗网格",
        "wave_speed": "波速",
        "damping": "阻尼",
        "source_shape": "源形状",
        "boundary": "边界",
        "non_gaussian_process": "非高斯过程噪声",
        "non_gaussian_observation": "非高斯观测噪声",
    }
    mismatch_rows = []
    for condition in completion["mismatch"]["conditions"]:
        selected = [item for item in completion["mismatch"]["summary"] if item["mismatch"] == condition and item["method"] in {"enkf", "A6_pce", "A7_apce"}]
        values = {item["method"]: item["mean_rmse"] for item in selected}
        mismatch_rows.append(f"{mismatch_labels[condition]} & {values['enkf']:.4e} & {values['A6_pce']:.4e} & {values['A7_apce']:.4e} \\\\"
        )
    (SUBMISSION / "table_mismatch.tex").write_text(
        "\n".join(mismatch_rows) + "\n\\bottomrule\n", encoding="utf-8"
    )


def main() -> None:
    for directory in (FIGURES, EXTENDED, SOURCE_DATA, SUBMISSION / "sections"):
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_framework_figure()
    make_main_wave_figure()
    make_ablation_figure()
    make_generalization_figure()
    make_multiphysics_figure()
    make_completion_figure()
    copy_extended_assets()
    copy_source_data()
    write_macros_and_tables()
    print(json.dumps({"submission": str(SUBMISSION), "figures": 6, "extended_assets": len(list(EXTENDED.iterdir()))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
