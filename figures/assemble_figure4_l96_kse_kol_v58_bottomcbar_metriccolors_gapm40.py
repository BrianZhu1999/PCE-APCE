from pathlib import Path

import assemble_figure4_l96_kse_kol_v56_nfix_pfix_gapm40 as assembly


HERE = Path(__file__).resolve().parent

assembly.KSE_PATH = HERE / "figure4_kse_mu11_two_rows_reconpatch_forecast_v8_u_below_attachment2.png"
assembly.KOL_PANEL = HERE / "figure4_kol_vorticity_reconstruction_1x4_v20_bottomcbar_midgap07_cbarh15_labelbottom.png"
assembly.KOL_FORECAST_ROW = HERE / "figure4_kol_forecast_panels_op_v5_bottomcbar_midgap07_metriccolors_cbarh15_labeltop.png"
assembly.OUTPUT_BASE = HERE / "figure4_reordered_l96_kse_kol_v58_bottomcbar_metriccolors_gapm40"


if __name__ == "__main__":
    assembly.main()
