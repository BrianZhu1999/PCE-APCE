# Paired Cumulative Predictive Evidence for Hybrid Uncertain Dynamics

This repository contains the source code for the paired cumulative predictive
evidence (PCE) and adaptive paired cumulative predictive evidence (APCE)
experiments accompanying the manuscript *Generative dynamic assimilation of
random--cognitive hybrid uncertain wave fields from sparse vector observations*.

The code implements a training-free ensemble assimilation workflow in which a
grid of candidate uncertainty coordinates is propagated by a forward model,
scored against observations, and combined through cumulative predictive
evidence. The repository also contains the case adapters and figure-generation
scripts used for the synthetic wave, applied ODE, high-dimensional chaotic,
VIV--PIV, MeshRIR and Baoding analyses.

## Scope of this release

This is a source-code release. Raw experimental data, private measurements,
large result trees, internal audit archives, manuscript source files and
compiled figures are intentionally kept out of the repository. The code is
therefore suitable for inspection, unit testing and rerunning experiments when
the corresponding data are obtained and paths are supplied locally.

The VIV--PIV dataset is publicly available at
[DOI 10.57745/HPA87O](https://doi.org/10.57745/HPA87O). The MeshRIR source
dataset is documented at [DOI 10.5281/zenodo.5002817](https://doi.org/10.5281/zenodo.5002817).
The Baoding measurements are not redistributed; the adapter accepts an
authorized local copy. The F-16 GVT pilot uses the public benchmark at
[DOI 10.4121/12954911.v1](https://doi.org/10.4121/12954911.v1).

## Repository map

| Path | Role |
| --- | --- |
| `hilda_da/` | Core PCE/APCE assimilation, evidence, flow, low-rank and metric modules |
| `experiments/` | Figure 2 runners, aggregation, gates and reproducibility utilities |
| `paper_experiments/` | Figure 3--5 experiment runners and post-processing |
| `viv_piv_case/` | Public VIV--PIV transfer case and Figure 5 source scripts |
| `meshir_case/` | MeshRIR/S1 reconstruction case |
| `baoding_case/` | Authorized-data Baoding tracking adapter |
| `f16_gvt_case/` | Isolated F-16 GVT pilot package |
| `figures/` | Source plotting and assembly scripts; generated media are ignored |
| `tests/` | Core, infrastructure and case-level tests |
| `docs/` | Reproduction notes and release boundaries |

## Installation

The supported baseline is Python 3.11. A conda environment is provided for a
repeatable scientific stack:

```bash
conda env create -f environment.yml
conda activate pce-apce-release
python -m pip install -e ".[dev,cases]"
```

For an existing environment:

```bash
python -m pip install -e ".[dev,cases]"
```

PyTorch wheels are platform-specific. Install the CUDA-matched wheel supplied
by the PyTorch project when GPU execution is required; the package itself does
not pin a CUDA runtime.

## Minimal smoke test

The following checks do not require paper result archives:

```bash
python -m compileall -q hilda_da experiments paper_experiments viv_piv_case meshir_case baoding_case f16_gvt_case
python -m pytest -q tests/test_metrics.py tests/test_systems.py tests/test_hilda_core.py
python run_hybrid_wave.py --mode quick --output results/quick_wave
```

The minimal test command assumes a working PyTorch installation. On systems
where the installed PyTorch wheel cannot load its native libraries, run the
non-PyTorch tests separately and install a platform-matched wheel before using
the assimilation modules.

Most full tests import PyTorch and case-specific scientific libraries. Run
them in the conda environment or on the Super-Server environment used for the
paper analyses.

## Running a public VIV--PIV case

Copy a configuration from `viv_piv_case/` and replace its path placeholders, or
set the two environment variables below:

```bash
export VIV_PIV_DATA_ROOT=/path/to/viv_piv_hpa87o
export VIV_PIV_OUTPUT_ROOT=/path/to/local/results/viv_piv
python -m viv_piv_case.audit --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json
python -m viv_piv_case.prepare --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json
python -m viv_piv_case.run_case --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json --case 0679 --method pce --seed 0 --record-trace
python -m viv_piv_case.aggregate --config viv_piv_case/config_adaptive_fullfield_x40y20_formal5.json
```

The configuration separates the twelve training cases from the five external
test cases. Cylinder displacement is supplied as a known exogenous input, and
the assimilation observation vector contains the selected velocity samples.
See `viv_piv_case/README.md` for the frozen protocol and data layout.

## Reproducing synthetic figures

Figure 2 and Figure 3 runners accept explicit result/output paths. The frozen
protocol matrices are included under `experiments/`, with machine-specific
locations replaced by repository-relative paths. Examples and the corresponding
command-line arguments are collected in `docs/reproduction.md`. Raw data,
scenario assets and formal result archives remain outside this code release.

## Data and privacy boundaries

Do not commit raw measurements, private Baoding data, remote workstation
paths, credentials, generated result archives or manuscript PDFs. Use local
paths or environment variables and keep generated outputs under an ignored
directory. The release records dataset identifiers and protocol metadata, not
the underlying restricted measurements.

## Citation and license

Please cite the accompanying manuscript and see `CITATION.cff` for machine-
readable metadata. The source code is released under the MIT License; see
`LICENSE`.
