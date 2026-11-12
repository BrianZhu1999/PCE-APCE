# PCE–APCE

**Forecast-history separation for dynamical inference**

Code and benchmark implementations accompanying *Observation feedback can conceal competing dynamics*. PCE and APCE separate observation-conditioned state reconstruction from candidate-dependent forecast scoring. Predictive weights connect the two histories; APCE additionally adapts evidence accumulation.

[Supplementary movies and interactive views](https://brianzhu1999.github.io/PCE-APCE/supplementary/) · [Reproduction guide](docs/reproduction.md) · [Citation](CITATION.cff)

## Repository contents

| Directory | Description |
| --- | --- |
| `pce_assimilation/` | Ensemble analysis and predictive weighting |
| `benchmarks/` | Nonlinear systems and high-dimensional dynamics |
| `viv_piv/` | Measured cylinder-wake reconstruction and forecasting |
| `acoustic_field_reconstruction/` | Three-dimensional acoustic-field reconstruction |
| `acoustic_array_tracking/` | Single- and dual-source acoustic tracking |
| `three_source_tracking/` | Three-source acoustic localization and tracking |
| `reproducibility/` | Controlled-feedback simulation and statistical reproduction |
| `docs/supplementary/` | Eight supplementary movies and five interactive views |
| `tests/` | Method and protocol tests |

## Installation

Python 3.11 is recommended.

```bash
conda env create -f environment.yml
conda activate pce-apce
python -m pip install -e ".[dev]"
python -m pytest -q
```

Install a compatible CUDA-enabled PyTorch build for GPU execution.

## Examples

```bash
python -m benchmarks.classical_systems --case wave --method apce --seed 2026080700 --output results/classical
python -m benchmarks.applied_odes --n-seeds 1 --device cpu --output results/applied_odes
```

See [reproduction instructions](docs/reproduction.md) and the README in each experiment directory for data preparation and commands.

## Data

- [VIV–PIV measured wakes](https://doi.org/10.57745/HPA87O)
- [MeshRIR acoustic measurements](https://doi.org/10.5281/zenodo.5002817)
- Acoustic tracking: local measured-array inputs specified in the experiment configuration

## Citation and licence

Citation metadata: [CITATION.cff](CITATION.cff). Source code: [MIT License](LICENSE).
