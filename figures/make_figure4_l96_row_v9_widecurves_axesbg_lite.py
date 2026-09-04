from __future__ import annotations

from pathlib import Path

import make_figure4_l96_row_v7_palette_attachment1 as src
from figure4_v59_style_helpers import L96_BACKGROUND_LITE, apply_v59_axes_style


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_l96_row_v9_widecurves_axesbg_lite"

_original_save_all = src.base.save_all


def save_all_v61(fig, output_base: Path):
    apply_v59_axes_style(fig, background=L96_BACKGROUND_LITE)
    return _original_save_all(fig, output_base)


src.base.save_all = save_all_v61


if __name__ == "__main__":
    src.main()
