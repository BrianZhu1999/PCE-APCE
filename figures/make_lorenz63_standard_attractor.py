from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "lorenz63_standard_attractor"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def lorenz63_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ],
        dtype=float,
    )


def integrate_lorenz63(
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.005,
    steps: int = 60000,
    discard: int = 6000,
) -> np.ndarray:
    states = np.empty((steps + 1, 3), dtype=float)
    states[0] = np.array([1.0, 1.0, 1.0], dtype=float)
    for idx in range(steps):
        state = states[idx]
        k1 = lorenz63_rhs(state, sigma, rho, beta)
        k2 = lorenz63_rhs(state + 0.5 * dt * k1, sigma, rho, beta)
        k3 = lorenz63_rhs(state + 0.5 * dt * k2, sigma, rho, beta)
        k4 = lorenz63_rhs(state + dt * k3, sigma, rho, beta)
        states[idx + 1] = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return states[discard:]


def style_3d(ax: plt.Axes) -> None:
    ax.view_init(elev=23.0, azim=-61.0)
    ax.grid(True, color="#D6D6D6", linewidth=0.45)
    ax.tick_params(labelsize=6.4, width=0.55, pad=0.0)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
        axis.pane.set_edgecolor("#DDDDDD")
        axis.line.set_color("#AFAFAF")
        axis.line.set_linewidth(0.55)


def make_figure() -> None:
    set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    states = integrate_lorenz63(sigma=sigma, rho=rho, beta=beta)
    source = OUTPUT / "lorenz63_standard_attractor_source.npz"
    np.savez_compressed(source, states=states, sigma=sigma, rho=rho, beta=beta, dt=0.005)

    fig = plt.figure(figsize=(4.8, 4.25))
    ax = fig.add_subplot(111, projection="3d")
    style_3d(ax)
    # Subsample only for vector-file size and visual smoothness; integration is dense.
    line = states[::2]
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#2B77B8", linewidth=0.45, alpha=0.92)
    ax.set_xlabel(r"$x$", labelpad=2.0)
    ax.set_ylabel(r"$y$", labelpad=2.0)
    ax.set_zlabel(r"$z$", labelpad=2.0)
    ax.set_xlim(-20, 22)
    ax.set_ylim(-28, 30)
    ax.set_zlim(0, 50)
    ax.text2D(
        0.5,
        1.045,
        r"Lorenz-63 attractor",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#242424",
    )
    ax.text2D(
        0.5,
        0.995,
        r"$\sigma=10,\ \rho=28,\ \beta=8/3$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#242424",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)
    base = OUTPUT / "Figure_lorenz63_standard_attractor"
    for suffix, kwargs in {".svg": {}, ".pdf": {}, ".png": {"dpi": 700}, ".tiff": {"dpi": 700}}.items():
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)

    svg = base.with_suffix(".svg").read_text(encoding="utf-8", errors="ignore")
    qa = {
        "deterministic": True,
        "process_noise": False,
        "parameters": {"sigma": sigma, "rho": rho, "beta": beta},
        "discarded_transient_steps": 6000,
        "no_hilda": "HILDA" not in svg,
        "no_chinese": not any("\u4e00" <= char <= "\u9fff" for char in svg),
        "no_bold": "font-weight:bold" not in svg and "font-weight:700" not in svg,
        "editable_svg_text": svg.count("<text") > 0,
    }
    (OUTPUT / "qa_manifest.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "source": str(source)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    make_figure()
