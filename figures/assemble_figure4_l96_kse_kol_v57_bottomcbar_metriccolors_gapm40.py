from pathlib import Path

import assemble_figure4_l96_kse_kol_v56_nfix_pfix_gapm40 as assembly


HERE = Path(__file__).resolve().parent

assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v19_bottomcbar_midgap07_cbarh15_labeltop.png"
assembly.KOL_FORECAST_ROW = HERE / "figure4_kol_forecast_panels_op_v5_bottomcbar_midgap07_metriccolors_cbarh15_labeltop.png"
assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v57_bottomcbar_metriccolors_gapm40"


if __name__ == "__main__":
    assembly.main()
