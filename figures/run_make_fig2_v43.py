from pathlib import Path
import subprocess
import sys

ROOT = Path(r".\hybrid_uncertain_wave")
FIG = ROOT / "figures"
SRC = ROOT / "ncs_chinese_submission" / "source_data" / "figure2_v2_main_20260810"
OUT = ROOT / "ncs_chinese_submission" / "figures"
REP = FIG / "figure2_v2_representative_source"
SEED = FIG / "figure2_v2_seedband_summary_20260810"

cmd = [
    sys.executable,
    str(FIG / "make_figure2_classical_systems_v39_v2_formal.py"),
    "--summary-csv", str(SRC / "figure2_v2_main_method_summary.csv"),
    "--runs-csv", str(SRC / "figure2_v2_main_run_source_data.csv"),
    "--paired-csv", str(SRC / "figure2_v2_main_paired_comparisons.csv"),
    "--wave-npz", str(REP / "wave_v2_representative_seed_2026080733.npz"),
    "--spring-npz", str(REP / "spring_v2_representative_seed_2026080739.npz"),
    "--heat-npz", str(REP / "heat_v2_representative_seed_2026080729.npz"),
    "--output-dir", str(OUT),
    "--output-stem", "figure2_classical_uncertain_systems_v54_calibration_sharpness_frontier",
]

subprocess.run(cmd, cwd=str(FIG), check=True)
