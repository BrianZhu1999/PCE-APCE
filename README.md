# PCE-APCE: Paired Cumulative Predictive Evidence

This repository implements paired cumulative predictive evidence (PCE) and
adaptive paired cumulative predictive evidence (APCE) for dynamic assimilation
under stochastic and cognitive uncertainty. Candidate dynamics are propagated
in parallel, evaluated through analysis-isolated shadow forecasts and combined
by cumulative predictive evidence.

The release contains the method, the benchmark protocols used in the paper and
three measured-data applications. Plotting code, generated figures, result
archives and exploratory studies are outside the public package.

## Installation

Python 3.11 is recommended.

```bash
conda env create -f environment.yml
conda activate pce-apce
python -m pip install -e ".[dev]"
python -m pytest -q
```

PyTorch should be installed with the CUDA build appropriate for the target
machine when GPU execution is required.

## Repository structure

| Path | Contents |
| --- | --- |
| `pce_assimilation/` | PCE/APCE evidence, assimilation and ensemble-analysis components |
| `benchmarks/` | Classical systems, applied ODEs and high-dimensional dynamics |
| `viv_piv/` | Sparse VIV-PIV wake reconstruction |
| `acoustic_field_reconstruction/` | Three-dimensional measured acoustic-field reconstruction |
| `acoustic_array_tracking/` | Single- and dual-source acoustic-array tracking |
| `tests/` | Focused method and protocol tests |

Each benchmark exposes a module entry point. For example:

```bash
python -m benchmarks.classical_systems --case wave --method apce \
  --seed 2026080700 --output results/classical
python -m benchmarks.applied_odes --n-seeds 1 --device cpu \
  --output results/applied_odes
```

The full command map and expected inputs are in
[`docs/reproduction.md`](docs/reproduction.md). Application-specific setup is
documented in the README inside each measured-data directory.

## Data

- VIV-PIV: [DOI 10.57745/HPA87O](https://doi.org/10.57745/HPA87O)
- MeshRIR: [DOI 10.5281/zenodo.5002817](https://doi.org/10.5281/zenodo.5002817)
- Acoustic-array tracking: the adapter reads an authorized local copy of the
  measured array data.

Data paths and output paths are explicit command-line or JSON configuration
values. Generated outputs are written under user-selected result directories
and are ignored by Git.

## Citation

Please cite the accompanying manuscript. Machine-readable software metadata is
provided in [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE).
