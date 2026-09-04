# Figure-to-source map

| Manuscript component | Source entry point |
| --- | --- |
| Figure 2, synthetic uncertainty systems | `figures/make_figure2_classical_systems_v39_v2_formal.py` and `experiments/` Figure 2 runners |
| Figure 3, applied ODE transfer | `paper_experiments/run_figure3_applied_ode.py` and `paper_experiments/make_figure3_main_v1.py` |
| Figure 4, high-dimensional systems | `paper_experiments/run_figure4_lorenz96_1024_scaling.py`, `paper_experiments/run_figure4_kolmogorov64_sparse2d.py`, and `figures/assemble_figure4_l96_kse_kol_v62_widecurves_axesbg_soft_olabelalign.py` |
| Figure 5, VIV--PIV transfer | `viv_piv_case/figure5_main/plot_figure5_viv_piv.py` and panel-specific scripts |
| MeshRIR/S1 reconstruction | `meshir_case/s1_reconstruction_192/run_stage_d.py` and `make_apce_3d_reconstruction_figure_v6.py` |

The code map identifies source entry points; it does not bundle the private
result archives used for the manuscript figures.
