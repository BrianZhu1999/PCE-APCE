from __future__ import annotations

from pathlib import Path

import assemble_figure4_l96_kse_kol_v18_vorticity_left as assembly


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v4_pixelmatched.png"
    assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v21_vorticity_pixelmatched_left"
    assembly.main()
