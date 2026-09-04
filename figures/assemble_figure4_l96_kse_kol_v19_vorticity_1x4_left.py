from __future__ import annotations

from pathlib import Path

import assemble_figure4_l96_kse_kol_v18_vorticity_left as assembly


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v2.png"
    assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v19_vorticity_1x4_left"
    assembly.main()
