from __future__ import annotations

from pathlib import Path

import make_figure4_kol_vorticity_reconstruction_1x4_v16_ref_label as src
from figure4_v59_style_helpers import KOL_BACKGROUND, apply_v59_axes_style, save_colored_face_exports


HERE = Path(__file__).resolve().parent
src.OUTPUT_BASE = HERE / "figure4_kol_vorticity_reconstruction_1x4_v21_systembg"


def save_all_v59(fig):
    apply_v59_axes_style(fig, background=KOL_BACKGROUND)
    return save_colored_face_exports(fig, src.OUTPUT_BASE, background=KOL_BACKGROUND, dpi=src.DPI)


src.save_all = save_all_v59


if __name__ == "__main__":
    src.main()
