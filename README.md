# PCE–APCE

## Paired cumulative predictive evidence for hybrid uncertain dynamics

PCE and APCE provide a common evidence update for dynamic assimilation when candidate models, observations and latent cognitive weights are uncertain. Candidate dynamics are propagated in parallel, scored with analysis-isolated shadow forecasts and combined through cumulative predictive evidence. APCE adds an entropy-aware update of the candidate mixture.

<p><a href="https://brianzhu1999.github.io/PCE-APCE/supplementary/"><strong>Open the supplementary showcase</strong></a> · <a href="docs/supplementary/README.md">Browse the release files</a> · <a href="docs/reproduction.md">Reproduce the experiments</a> · <a href="CITATION.cff">Cite this software</a></p>

The repository pairs the implementation with the benchmark procedures used in the accompanying manuscript and a compact, reader-facing collection of eight supplementary movies and five interactive HTML viewers. The media gallery is designed for inspection: each item states its case, purpose and evaluation setting, while the public catalogue records checksums and source details.

## What is here

| Path | Contents |
| --- | --- |
| `pce_assimilation/` | PCE/APCE evidence updates, assimilation and ensemble analysis |
| `benchmarks/` | Classical systems, applied ODEs and high-dimensional dynamics |
| `viv_piv/` | Sparse VIV–PIV wake reconstruction |
| `acoustic_field_reconstruction/` | Three-dimensional measured acoustic-field reconstruction |
| `acoustic_array_tracking/` | Single- and dual-source acoustic-array tracking |
| `docs/supplementary/` | Movies, interactive viewers and the public media catalogue |
| `tests/` | Focused method and experiment-setting tests |

The supplementary collection is an explanatory companion to the code, with reader-facing movies and case-specific interactive viewers.

## Installation

Python 3.11 is recommended.

```bash
conda env create -f environment.yml
conda activate pce-apce
python -m pip install -e ".[dev]"
python -m pytest -q
```

Install the PyTorch CUDA build appropriate for the target machine when GPU execution is required. CPU execution is sufficient for the small examples.

## Reproduce

Each benchmark exposes a module entry point. For example:

```bash
python -m benchmarks.classical_systems --case wave --method apce \
  --seed 2026080700 --output results/classical
python -m benchmarks.applied_odes --n-seeds 1 --device cpu \
  --output results/applied_odes
```

The complete command map, expected inputs and case-specific settings are in [`docs/reproduction.md`](docs/reproduction.md). Measured-data applications document their preparation steps in the README inside each directory.

## Data and media

- VIV–PIV: [DOI 10.57745/HPA87O](https://doi.org/10.57745/HPA87O)
- MeshRIR: [DOI 10.5281/zenodo.5002817](https://doi.org/10.5281/zenodo.5002817)
- Acoustic-array tracking: the adapter reads an authorized local copy of the measured array data; the measurements are not embedded in this repository.

Data and output locations are explicit command-line or JSON configuration values. Generated outputs are written to user-selected result directories and remain outside version control unless selected for the supplementary gallery. See [`docs/supplementary/README.md`](docs/supplementary/README.md) for the movie and viewer catalogue, integrity records and local viewing instructions.

## Citation

Please cite the accompanying manuscript and this software release. Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

The source code is released under the [MIT License](LICENSE). The supplementary media remain part of this project release and retain their source/data attribution; no additional media licence is asserted here.
