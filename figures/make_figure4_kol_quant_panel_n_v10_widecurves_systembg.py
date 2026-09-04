from __future__ import annotations

from pathlib import Path

import make_figure4_kol_quant_panel_n_v8_attachment3_boxes as src
from figure4_v59_style_helpers import KOL_BACKGROUND, apply_v59_axes_style, save_transparent_exports


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_kol_quant_panel_n_v10_widecurves_systembg"


def save_all_v59(fig):
    apply_v59_axes_style(fig, background=KOL_BACKGROUND)
    return save_transparent_exports(fig, src.OUTPUT_BASE, dpi=src.DPI)


src.save_all = save_all_v59


if __name__ == "__main__":
    src.main()
