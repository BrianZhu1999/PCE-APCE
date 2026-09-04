from __future__ import annotations

from pathlib import Path

import assemble_figure4_l96_kse_kol_v18_vorticity_left as kol_assembly
import assemble_figure4_reordered_caseboxed_v8 as base


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    old_kse = base.KSE_PATH
    old_master = (base.OUT_PNG, base.OUT_PDF, base.OUT_TIFF)
    try:
        base.KSE_PATH = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v6_relabel_gl_titles4p08_chrome.png"
        base.OUT_PNG = HERE / "figure4_reordered_l96_kse_v18_title20_kse_titles4p08_chrome.png"
        base.OUT_PDF = HERE / "figure4_reordered_l96_kse_v18_title20_kse_titles4p08_chrome.pdf"
        base.OUT_TIFF = HERE / "figure4_reordered_l96_kse_v18_title20_kse_titles4p08_chrome.tiff"
        base.main()
    finally:
        base.KSE_PATH = old_kse
        base.OUT_PNG, base.OUT_PDF, base.OUT_TIFF = old_master

    old_input = kol_assembly.INPUT_PNG
    old_panel = kol_assembly.KOL_PANEL
    old_output = kol_assembly.OUTPUT_BASE
    old_x = kol_assembly.KOL_PANEL_X
    try:
        kol_assembly.INPUT_PNG = HERE / "figure4_reordered_l96_kse_v18_title20_kse_titles4p08_chrome.png"
        kol_assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v12_titlegap_up_tlabels.png"
        kol_assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v34_kse_titles4p08_chrome"
        kol_assembly.KOL_PANEL_X = 3
        kol_assembly.main()
    finally:
        kol_assembly.INPUT_PNG = old_input
        kol_assembly.KOL_PANEL = old_panel
        kol_assembly.OUTPUT_BASE = old_output
        kol_assembly.KOL_PANEL_X = old_x
