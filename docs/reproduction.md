# Reproduction notes

This document describes the interfaces needed to rerun the code with local
data. It does not contain the private result archives used to assemble the
manuscript.

## Core wave experiment

From the repository root:

```bash
python run_hybrid_wave.py --mode quick --filter lr --output results/wave_quick
```

The command writes metrics and optional diagnostic arrays to the requested
output directory. Use a separate output directory for every seed or protocol.

## Figure 2 experiment runners

The scripts under `experiments/` expose explicit `--output-root` or
`--result-root` arguments. Keep these paths outside tracked source folders.
Run a small local check before a matrix:

```bash
python experiments/run_cpu_compatibility_smoke.py --help
python experiments/run_assimilation.py --help
python experiments/aggregate_figure2_formal.py --help
```

The protocol matrices are included in `experiments/`; their data and output
locations are repository-relative examples. The formal Figure 2 runs also
require the frozen scenario assets, which are not included. The source scripts
preserve the candidate grid, observation protocol and aggregation definitions.

## Applied ODE and high-dimensional cases

Use the `paper_experiments/` runners with explicit input and output paths.
Data adapters for external PDE/chaotic benchmarks expect arrays supplied by
the user; no external dataset is silently downloaded.

## VIV--PIV

Set `VIV_PIV_DATA_ROOT` and `VIV_PIV_OUTPUT_ROOT`, or edit a copied JSON
configuration. The public archive is identified by DOI 10.57745/HPA87O. A
typical single-case sequence is:

```bash
python -m viv_piv_case.audit --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json
python -m viv_piv_case.prepare --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json
python -m viv_piv_case.run_case --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json --case 0679 --method apce --seed 0 --record-trace
python -m viv_piv_case.aggregate --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json
```

## MeshRIR and Baoding

MeshRIR scripts accept an explicitly supplied dataset/cache path; obtain the
dataset from DOI 10.5281/zenodo.5002817. Baoding scripts require an authorized
local data root and never ship the measurements. The command-line interfaces
use `--remote-root`, `--input-root` or `--data-root` for this reason.

## Provenance

Record the dataset DOI/version, configuration file, seed, software versions
and output directory for each run. Generated manifests and result files are
local artifacts and are excluded by `.gitignore`.
