from __future__ import annotations

import json
from pathlib import Path

import assemble_figure4_l96_kse_kol_v56_nfix_pfix_gapm40 as assembly
from figure4_v62_style_helpers import (
    KOL_BACKGROUND_SOFT,
    KSE_BACKGROUND_SOFT,
    L96_BACKGROUND_SOFT,
    LINEWIDTH_FACTOR,
)


HERE = Path(__file__).resolve().parent

assembly.L96_PATH = HERE / "figure4_l96_row_v10_widecurves_axesbg_soft.png"
assembly.KSE_PATH = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v12_widecurves_axesbg_soft.png"
assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v24_axesbg_soft.png"
assembly.KOL_PANEL_N = HERE / "figure4_kol_quant_panel_n_v13_widecurves_axesbg_soft.png"
assembly.KOL_FORECAST_ROW = HERE / "figure4_kol_forecast_panels_op_v9_widecurves_axesbg_soft_olabelalign.png"
assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v62_widecurves_axesbg_soft_olabelalign"


if __name__ == "__main__":
    assembly.main()
    qa_path = assembly.OUTPUT_BASE.with_suffix(".qa.json")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa.update(
        {
            "current_composite": "figure4_reordered_l96_kse_kol_v62_widecurves_axesbg_soft_olabelalign",
            "preserved_previous_composite": "figure4_reordered_l96_kse_kol_v61_widecurves_axesbg_lite_olabelalign",
            "preserved_v58_composite": "figure4_reordered_l96_kse_kol_v58_bottomcbar_metriccolors_gapm40",
            "visual_update_only": True,
            "v62_style": {
                "linewidth_factor": LINEWIDTH_FACTOR,
                "system_backgrounds": {
                    "lorenz96_axes": L96_BACKGROUND_SOFT,
                    "kuramoto_sivashinsky_axes": KSE_BACKGROUND_SOFT,
                    "kolmogorov_axes": KOL_BACKGROUND_SOFT,
                },
                "background_scope": "axes only; no full-row/system band fill",
                "background_intent": "slightly stronger pastel system tint than v61 to reduce the near-white feel without competing with data",
                "pane_alpha": 0.65,
                "panel_o_alignment": "panel o source-label x is compensated to match panel m after KOL_PANEL_X=3 px in the composite",
                "scientific_data_changed": False,
            },
            "source_scripts": {
                "lorenz96": str(HERE / "make_figure4_l96_row_v10_widecurves_axesbg_soft.py"),
                "kse": str(HERE / "make_figure4_kse_mu11_two_rows_reconpatch_forecast_v12_widecurves_axesbg_soft.py"),
                "kol_m": str(HERE / "make_figure4_kol_vorticity_reconstruction_1x4_v24_axesbg_soft.py"),
                "kol_n": str(HERE / "make_figure4_kol_quant_panel_n_v13_widecurves_axesbg_soft.py"),
                "kol_op": str(HERE / "make_figure4_kol_forecast_panels_op_v9_widecurves_axesbg_soft_olabelalign.py"),
                "assembly": str(Path(__file__).resolve()),
            },
            "remote_source_roots": {
                "kse_reconstruction": "<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi32_t2_core4_formal_50seeds_20260814_4gpu",
                "kse_forecast": "<HILDA_RESULTS_ROOT>/results/figure5_kse_blackout32_t2_step40_100seeds_20260814_4gpu",
                "kse_reconstruction_patch": "<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi32_t2_seed90_sample00_reconstruction_patch_20260814",
            },
        }
    )
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
