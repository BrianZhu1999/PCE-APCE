from __future__ import annotations

from pathlib import Path

import make_figure4_kol_vorticity_reconstruction_1x4_v16_ref_label as src
from figure4_v62_style_helpers import (
    KOL_BACKGROUND_SOFT,
    apply_v62_axes_style,
    save_axes_background_with_transparent_figure,
)


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_kol_vorticity_reconstruction_1x4_v24_axesbg_soft"


def save_all_v62(fig):
    apply_v62_axes_style(fig, background=KOL_BACKGROUND_SOFT)
    return save_axes_background_with_transparent_figure(fig, src.OUTPUT_BASE, dpi=src.DPI)


src.save_all = save_all_v62


if __name__ == "__main__":
    src.main()
