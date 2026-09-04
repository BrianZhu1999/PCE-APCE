from __future__ import annotations

import sys
from pathlib import Path

import make_figure4_kse_mu11_two_rows_reconpatch_forecast_v7_i_palette_attachment2 as src
from figure4_v59_style_helpers import KSE_BACKGROUND, apply_v59_axes_style, save_colored_face_exports


HERE = Path(__file__).resolve().parent
OUTPUT_BASE = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v9_widecurves_systembg"

_original_save_all = src.save_all


def save_all_v59(fig, output_base: Path) -> None:
    apply_v59_axes_style(fig, background=KSE_BACKGROUND)
    save_colored_face_exports(
        fig,
        output_base,
        background=KSE_BACKGROUND,
        dpi=650,
        bbox_inches="tight",
        pad_inches=0.03,
    )


src.save_all = save_all_v59


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--output", str(OUTPUT_BASE)]
    src.main()
