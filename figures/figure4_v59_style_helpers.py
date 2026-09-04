from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt


LINEWIDTH_FACTOR = 1.25

L96_BACKGROUND = "#F7F6FF"
KSE_BACKGROUND = "#F4FAF8"
KOL_BACKGROUND = "#FFF8F1"

L96_BACKGROUND_LITE = "#FCFBFF"
KSE_BACKGROUND_LITE = "#FBFEFD"
KOL_BACKGROUND_LITE = "#FFFCF8"


def apply_v59_axes_style(
    fig: plt.Figure,
    *,
    background: str,
    linewidth_factor: float = LINEWIDTH_FACTOR,
) -> None:
    """Apply the Figure 4 v59 visual-only style before export.

    This routine does not alter data, axis ranges, labels, legends or
    selection logic. It only gives each axes a pale system-level background
    and multiplies existing line/box/spine widths.
    """
    for ax in fig.axes:
        ax.set_facecolor(background)

        for line in ax.lines:
            try:
                current = float(line.get_linewidth())
                if current > 0:
                    line.set_linewidth(current * linewidth_factor)
            except Exception:
                pass

        for patch in ax.patches:
            try:
                current = float(patch.get_linewidth())
                if current > 0:
                    patch.set_linewidth(current * linewidth_factor)
            except Exception:
                pass

        for spine in ax.spines.values():
            try:
                current = float(spine.get_linewidth())
                if current > 0:
                    spine.set_linewidth(current * linewidth_factor)
            except Exception:
                pass

        # 3D axes store grid panes separately; the pale background should be
        # visible without changing the trajectory coordinates or colors.
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, axis_name, None)
            pane = getattr(axis, "pane", None)
            if pane is not None:
                try:
                    pane.set_facecolor(mpl.colors.to_rgba(background, 0.55))
                    pane.set_edgecolor("#E6E1ED")
                except Exception:
                    pass


def save_transparent_exports(fig: plt.Figure, output_base: Path, *, dpi: int = 650) -> dict[str, str]:
    outputs = {
        "png": output_base.with_suffix(".png"),
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
        "tiff": output_base.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=dpi, transparent=True)
    fig.savefig(outputs["pdf"], transparent=True)
    fig.savefig(outputs["svg"], transparent=True)
    fig.savefig(outputs["tiff"], dpi=dpi, transparent=True)
    return {name: str(path) for name, path in outputs.items()}


def save_axes_background_with_transparent_figure(
    fig: plt.Figure,
    output_base: Path,
    *,
    dpi: int = 650,
    bbox_inches: str | None = None,
    pad_inches: float = 0.0,
) -> dict[str, str]:
    """Save with transparent figure background but opaque axes backgrounds.

    Matplotlib's ``transparent=True`` also makes axes patches transparent, so
    use an alpha-zero figure patch and a normal save instead. This keeps the
    v59/v60 system tint confined to small axes rather than filling the whole
    row image.
    """
    outputs = {
        "png": output_base.with_suffix(".png"),
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
        "tiff": output_base.with_suffix(".tiff"),
    }
    fig.patch.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0.0)
    kwargs: dict[str, Any] = {"facecolor": fig.get_facecolor(), "edgecolor": "none"}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
        kwargs["pad_inches"] = pad_inches
    fig.savefig(outputs["png"], dpi=dpi, transparent=False, **kwargs)
    fig.savefig(outputs["pdf"], transparent=False, **kwargs)
    fig.savefig(outputs["svg"], transparent=False, **kwargs)
    fig.savefig(outputs["tiff"], dpi=dpi, transparent=False, **kwargs)
    return {name: str(path) for name, path in outputs.items()}


def save_colored_face_exports(
    fig: plt.Figure,
    output_base: Path,
    *,
    background: str,
    dpi: int = 650,
    bbox_inches: str | None = None,
    pad_inches: float = 0.0,
) -> dict[str, str]:
    outputs = {
        "png": output_base.with_suffix(".png"),
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
        "tiff": output_base.with_suffix(".tiff"),
    }
    fig.patch.set_facecolor(background)
    fig.patch.set_alpha(1.0)
    kwargs: dict[str, Any] = {"facecolor": background}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
        kwargs["pad_inches"] = pad_inches
    fig.savefig(outputs["png"], dpi=dpi, **kwargs)
    fig.savefig(outputs["pdf"], **kwargs)
    fig.savefig(outputs["svg"], **kwargs)
    fig.savefig(outputs["tiff"], dpi=dpi, **kwargs)
    return {name: str(path) for name, path in outputs.items()}
