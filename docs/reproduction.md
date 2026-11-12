# Reproduction guide

Install the dependencies using the commands in the [repository README](../README.md). Experiment-specific settings are stored in the corresponding configuration files.

## Numerical benchmarks

```bash
python -m benchmarks.classical_systems --case wave --method apce --seed 2026080700 --output results/classical
python -m benchmarks.applied_odes --n-seeds 1 --device cpu --output results/applied_odes
python -m benchmarks.lorenz96 --help
python -m benchmarks.kuramoto_sivashinsky --help
python -m benchmarks.kolmogorov_velocity --help
python -m benchmarks.kolmogorov_blackout --help
```

High-dimensional flow cases require their benchmark datasets.

## Measured cylinder wakes

Download the [VIV–PIV dataset](https://doi.org/10.57745/HPA87O), set the data and output directories in a copy of `viv_piv/protocol.json`, and run:

```bash
python -m viv_piv.validate_data --config path/to/protocol.json
python -m viv_piv.prepare --config path/to/protocol.json
python -m viv_piv.prepare_adaptive_sensor_layout --config path/to/protocol.json --layout adaptive_fullfield_valid_x40y20
python -m viv_piv.prepare_observation_covariance --config path/to/protocol.json --layout adaptive_fullfield_valid_x40y20 --device cuda:0
python -m viv_piv.run_case --config path/to/protocol.json --case 0679 --method apce --seed 0 --record-trace
python -m viv_piv.aggregate --config path/to/protocol.json
```

Training and held-out regimes are specified in [viv_piv/README.md](../viv_piv/README.md).

## Three-dimensional acoustic field

Prepare a MeshRIR S1 dataset input and run:

```bash
python -m acoustic_field_reconstruction.run --source-npz path/to/s1_reconstruction_input.npz --output results/acoustic_field --device cuda:0
```

See [acoustic_field_reconstruction/](../acoustic_field_reconstruction/) for the protocol.

## Acoustic-source tracking

For single- and dual-source experiments, see [acoustic_array_tracking/](../acoustic_array_tracking/). For three-source experiments, see [three_source_tracking/](../three_source_tracking/). Acoustic measurements and reference tracks are specified through input paths.

## Controlled feedback and statistical checks

See [reproducibility/](../reproducibility/) for the scalar feedback simulation and the matched-history statistical analysis scripts.
