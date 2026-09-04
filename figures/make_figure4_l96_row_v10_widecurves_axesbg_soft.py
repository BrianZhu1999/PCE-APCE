from __future__ import annotations

from pathlib import Path

import make_figure4_l96_row_v7_palette_attachment1 as src
from figure4_v62_style_helpers import L96_BACKGROUND_SOFT, apply_v62_axes_style


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_l96_row_v10_widecurves_axesbg_soft"

_original_save_all = src.base.save_all


def save_all_v62(fig, output_base: Path):
    apply_v62_axes_style(fig, background=L96_BACKGROUND_SOFT)
    return _original_save_all(fig, output_base)


src.base.save_all = save_all_v62


if __name__ == "__main__":
    src.main()
