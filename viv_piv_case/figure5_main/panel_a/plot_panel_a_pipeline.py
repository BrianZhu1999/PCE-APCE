"""Render Figure 5a as the corrected x40-y20 VIV-PIV processing pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs_pipeline_x40y20"
OUT.mkdir(parents=True, exist_ok=True)
STEM = "figure5a_viv_piv_pipeline_x40y20"
DPI = 650
FIG_W_PX, FIG_H_PX = 10553, 2520
FIG_W, FIG_H = FIG_W_PX / DPI, FIG_H_PX / DPI

BLACK = "#202020"
GRAY = "#7F8C8D"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
PALE_GRAY = "#F3F5F5"
PALE_BLUE = "#EEF4F9"
PALE_ORANGE = "#FFF4E7"
PALE_GREEN = "#EEF6F2"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str,
        face: str, edge: str, *, title_color: str = BLACK, body_color: str = BLACK,
        dashed: bool = False, title_size: int = 14, body_size: int = 9) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
        linestyle="--" if dashed else "-",
    ))
    ax.text(x + w / 2, y + h - 0.24 * h, title, ha="center", va="center",
            fontsize=title_size, color=title_color)
    ax.text(x + w / 2, y + 0.38 * h, body, ha="center", va="center",
            fontsize=body_size, color=body_color, linespacing=1.25)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = GRAY, *, dashed: bool = False, rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.15,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    ))


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W_PX)
    ax.set_ylim(0, FIG_H_PX)
    ax.axis("off")

    # Current Figure 5 typography: panel 22 pt, title 14 pt, labels 13 pt,
    # compact annotations 11 pt.
    ax.text(92, 2390, "a", fontsize=22, fontweight="bold", va="center", color=BLACK)
    ax.text(190, 2390, "VIV–PIV assimilation and blackout-evaluation pipeline",
            fontsize=14, va="center", color=BLACK)

    # Lane labels make the information boundary explicit without adding a
    # separate legend or a long explanatory caption inside the panel.
    ax.text(180, 1910, "OFFLINE TRAINING", fontsize=11, color=GRAY, va="center")
    ax.text(180, 1130, "ONLINE HELD-OUT TEST", fontsize=11, color=BLUE, va="center")

    # Shared data source and strict split.
    box(ax, 180, 1425, 1380, 430, "Real VIV–PIV data",
        "201 × 416 velocity grid\n12 train + 5 held-out\ny_c(t) control input",
        PALE_GRAY, GRAY, title_color=BLACK)
    arrow(ax, (1565, 1680), (1810, 1850), GRAY, rad=0.14)
    arrow(ax, (1565, 1580), (1810, 1080), BLUE, rad=-0.14)

    # Offline lane.
    box(ax, 1810, 1760, 1450, 360, "Training cases only",
        "12 regimes\nmean + latent z_t", PALE_GRAY, GRAY)
    box(ax, 3500, 1760, 1250, 360, "POD",
        "rank = 256\nlatent state z_t", PALE_BLUE, BLUE, title_color=BLUE)
    box(ax, 4790, 1760, 1450, 360, "DMDc candidate library",
        "12 candidate regimes\ncontrolled dynamics", PALE_ORANGE, ORANGE, title_color=ORANGE)
    box(ax, 6470, 1760, 1500, 360, "Training-only calibration",
        "noise + covariance\nPCE/APCE settings", PALE_GRAY, GRAY)
    arrow(ax, (3260, 1940), (3490, 1940), GRAY)
    arrow(ax, (4755, 1940), (4780, 1940), GRAY)
    arrow(ax, (6245, 1940), (6460, 1940), GRAY)

    # Online lane.
    box(ax, 1810, 900, 1450, 430, "Held-out test cases",
        "U_r = 4.63, 5.56, 6.79, 8.03, 13.59\nstrictly held out", PALE_BLUE, BLUE, title_color=BLUE)
    box(ax, 3500, 900, 1250, 430, "Mask-aware sparse PIV",
        "x40 × y20 lattice\n800 → 751 valid\n1,502 u/v scalars", PALE_BLUE, BLUE, title_color=BLUE)
    box(ax, 4790, 900, 1450, 430, "Candidate forecasts",
        "PCE / APCE branches\ncomparison methods", PALE_GRAY, GRAY)
    arrow(ax, (3260, 1115), (3490, 1115), BLUE)
    arrow(ax, (4755, 1115), (4780, 1115), BLUE)

    # Shadow-analysis split.
    box(ax, 6470, 1110, 1400, 330, "Shadow branch",
        "evidence → weights", PALE_ORANGE, ORANGE,
        title_color=ORANGE, dashed=True)
    box(ax, 6470, 720, 1400, 330, "Analysis branch",
        "PIV update → mixture\nfull-field state", PALE_BLUE, BLUE, title_color=BLUE)
    arrow(ax, (6245, 1120), (6455, 1275), ORANGE, rad=0.10)
    arrow(ax, (6245, 1030), (6455, 885), BLUE, rad=-0.10)
    ax.text(7170, 675, "shadow does not receive analysis update", fontsize=11,
            color=ORANGE, ha="center", va="center")

    # Blackout transition and forecast horizons.
    box(ax, 8140, 900, 1370, 540, "Observation blackout",
        "PIV off at t_b\nweights frozen\ny_c(t) retained\nconditional forecast", PALE_ORANGE, ORANGE, title_color=ORANGE)
    arrow(ax, (7880, 885), (8120, 1035), BLUE, rad=0.08)
    arrow(ax, (7880, 1275), (8120, 1240), ORANGE, rad=-0.05)
    for idx, label in enumerate(("0.5 s", "1 s", "2 s", "4 s")):
        x = 8400 + idx * 245
        ax.plot(x, 790, marker="o", ms=5, color=ORANGE, mec="white", mew=0.5)
        ax.text(x, 700, label, fontsize=11, color=ORANGE, ha="center", va="center")

    # Final evaluation block and link to the existing b-j panels.
    box(ax, 9680, 1425, 700, 430, "Formal evaluation",
        "5 cases × 4 methods\n× 5 seeds\n100 formal runs", PALE_GREEN, GRAY)
    arrow(ax, (9515, 1150), (9660, 1550), GRAY, rad=-0.10)
    ax.text(10030, 1235, "b–j", fontsize=14, color=BLACK, ha="center", va="center")
    ax.text(10030, 1120, "fields · errors · spectra\nSt · PDFs · evidence", fontsize=11,
            color=GRAY, ha="center", va="center", linespacing=1.25)

    # A restrained footer closes the causal chain without repeating numerical
    # results that are already shown in panels f-j.
    ax.text(5265, 260, "sparse observations → shadow evidence + analysis update → blackout forecast → diagnostics",
            fontsize=10, color=BLACK, ha="center", va="center")

    outputs = {}
    stem = OUT / STEM
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, facecolor="white", pad_inches=0)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI, facecolor="white", pad_inches=0)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", pad_inches=0)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", pad_inches=0)
    plt.close(fig)
    outputs = {ext: str(stem.with_suffix(ext)) for ext in (".png", ".tiff", ".pdf", ".svg")}
    metadata = {
        "figure": STEM,
        "panel": "a",
        "core_conclusion": "The VIV-PIV experiment uses a training-only reduced-order candidate library, x40-y20 mask-aware sparse observations and separated shadow/analysis branches to test conditional full-field forecasting after PIV blackout.",
        "backend": "Python/matplotlib only",
        "canvas": {"width_px": FIG_W_PX, "height_px": FIG_H_PX, "dpi": DPI},
        "font_sizes_pt": {"panel": 22, "title": 14, "axis": 13, "compact": 11},
        "observation_layout": {
            "name": "adaptive_fullfield_valid_x40y20",
            "x_points": 40,
            "y_points": 20,
            "nominal_points": 800,
            "effective_points": 751,
            "scalar_observations": 1502,
            "mask_aware": True,
        },
        "protocol": {
            "train_cases": 12,
            "held_out_cases": [4.63, 5.56, 6.79, 8.03, 13.59],
            "pod_rank": 256,
            "candidate_models": 12,
            "ensemble_size": 64,
            "formal_seeds": 5,
            "methods": ["PCE", "APCE", "Aug-EnKF", "BMA"],
            "blackout_horizons_s": [0.5, 1.0, 2.0, 4.0],
        },
        "integrity": {
            "test_cases_used_for_model_fit": False,
            "shadow_receives_analysis_update": False,
            "weights_interpretation": "operational predictive evidence, not Bayesian posterior probabilities",
        },
        "sources": {
            "authoritative_remote_result_root": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5",
            "local_layout_manifest": str(ROOT.parent / "results_preview" / "x40y20_formal5" / "excel_source" / "manifest.json"),
            "local_source_data": str(ROOT / "source_data" / "figure5_viv_piv_compact_source_x40y20.npz"),
        },
        "outputs": outputs,
    }
    metadata["output_sha256"] = {ext: sha256(stem.with_suffix(ext)) for ext in outputs}
    (OUT / f"{STEM}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
