# Three-dimensional acoustic-field reconstruction

This package reconstructs the measured MeshRIR S1 field on a 21 by 21 by 9
grid. Geometry-only farthest-point sampling selects 128 boundary microphones
and 64 interior microphones. The remaining 2,463 interior points form the
reconstruction set; a separate 1,314-point boundary set evaluates boundary
completion.

Dataset: [DOI 10.5281/zenodo.5002817](https://doi.org/10.5281/zenodo.5002817).

Eight boundary-closure candidates are constructed from the measured boundary
subset. Their predicted fields drive PCE/APCE evidence, while the 64 interior
measurements update the wave state. The protocol, finite-difference settings
and assimilation parameters are stored in [`protocol.json`](protocol.json).

```bash
python -m acoustic_field_reconstruction.run \
  --source-npz path/to/s1_reconstruction_input.npz \
  --output results/acoustic_field --device cuda:2
```
