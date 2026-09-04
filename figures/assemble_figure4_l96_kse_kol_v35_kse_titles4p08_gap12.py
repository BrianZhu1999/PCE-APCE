from __future__ import annotations

from pathlib import Path

import assemble_figure4_l96_kse_kol_v18_vorticity_left as kol_assembly
import assemble_figure4_reordered_caseboxed_v8 as base


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    old_kse = base.KSE_PATH
    old_master = (base.OUT_PNG, base.OUT_PDF, base.OUT_TIFF)
    old_gap = base.KSE_INNER_GAP
    try:
        base.KSE_PATH = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_titles4p08_chrome.png"
        base.KSE_INNER_GAP = 12
        base.OUT_PNG = HERE / "figure4_reordered_l96_kse_v19_title20_kse_titles4p08_gap12.png"
        base.OUT_PDF = HERE / "figure4_reordered_l96_kse_v19_title20_kse_titles4p08_gap12.pdf"
        base.OUT_TIFF = HERE / "figure4_reordered_l96_kse_v19_title20_kse_titles4p08_gap12.tiff"
        base.main()
    finally:
        base.KSE_PATH = old_kse
        base.OUT_PNG, base.OUT_PDF, base.OUT_TIFF = old_master
        base.KSE_INNER_GAP = old_gap

    old_input = kol_assembly.INPUT_PNG
    old_panel = kol_assembly.KOL_PANEL
    old_output = kol_assembly.OUTPUT_BASE
    old_x = kol_assembly.KOL_PANEL_X
    try:
        kol_assembly.INPUT_PNG = HERE / "figure4_reordered_l96_kse_v19_title20_kse_titles4p08_gap12.png"
        kol_assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v12_titlegap_up_tlabels.png"
        kol_assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v35_kse_titles4p08_gap12"
        kol_assembly.KOL_PANEL_X = 3
        kol_assembly.main()
    finally:
        kol_assembly.INPUT_PNG = old_input
        kol_assembly.KOL_PANEL = old_panel
        kol_assembly.OUTPUT_BASE = old_output
        kol_assembly.KOL_PANEL_X = old_x
