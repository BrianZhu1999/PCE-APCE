from __future__ import annotations

from pathlib import Path

import make_figure4_kol_quant_panel_n_v8_attachment3_boxes as src
from figure4_v59_style_helpers import (
    KOL_BACKGROUND_LITE,
    apply_v59_axes_style,
    save_axes_background_with_transparent_figure,
)


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_kol_quant_panel_n_v12_widecurves_axesbg_lite"


def save_all_v61(fig):
    apply_v59_axes_style(fig, background=KOL_BACKGROUND_LITE)
    return save_axes_background_with_transparent_figure(fig, src.OUTPUT_BASE, dpi=src.DPI)


src.save_all = save_all_v61


if __name__ == "__main__":
    src.main()
