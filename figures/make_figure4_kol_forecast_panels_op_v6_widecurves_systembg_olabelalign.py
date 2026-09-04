from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

import make_figure4_kol_forecast_panels_op_v1 as src
from figure4_v59_style_helpers import KOL_BACKGROUND, apply_v59_axes_style, save_transparent_exports


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_kol_forecast_panels_op_v6_widecurves_systembg_olabelalign"

# Align panel o to the final x-position of panel m after the m row is placed
# with KOL_PANEL_X=3 px in the total Figure 4 assembly.
M_SOURCE_W_PX = 5947
KOL_PANEL_X_PX = 3
O_LABEL_X = (0.012 * M_SOURCE_W_PX + KOL_PANEL_X_PX) / src.FIG_W_PX

_original_figure_text = Figure.text


def figure_text_v59(self, x, y, s, *args, **kwargs):
    if s == "o" and abs(float(x) - 0.012) < 1.0e-9 and kwargs.get("fontweight") == "bold":
        x = O_LABEL_X
    return _original_figure_text(self, x, y, s, *args, **kwargs)


def save_all_v59(fig):
    apply_v59_axes_style(fig, background=KOL_BACKGROUND)
    return save_transparent_exports(fig, src.OUTPUT_BASE, dpi=src.DPI)


src.save_all = save_all_v59


if __name__ == "__main__":
    Figure.text = figure_text_v59
    try:
        src.main()
    finally:
        Figure.text = _original_figure_text
