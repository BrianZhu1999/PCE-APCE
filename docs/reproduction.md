# Reproducing the experiments

All runners accept explicit input and output paths. Start with one seed, then
use the same command with the paper seed range on a suitable compute node.

## Synthetic systems

```bash
# Wave, nonlinear spring and heat equation
python -m benchmarks.classical_systems --case wave --method apce \
  --seed 2026080700 --output results/classical

# Chemical reaction, pharmacokinetic infusion, pendulum,
# FitzHugh-Nagumo and Robertson kinetics
python -m benchmarks.applied_odes --n-seeds 1 --device cpu \
  --output results/applied_odes

# High-dimensional dynamics
python -m benchmarks.lorenz96 --help
python -m benchmarks.kuramoto_sivashinsky --help
python -m benchmarks.kolmogorov_velocity --help
python -m benchmarks.kolmogorov_blackout --help
```

The classical and applied-ODE runners generate paired truth, observation and
forecast perturbations from the requested seed. Kolmogorov-flow runners require
the corresponding local data path.

## VIV-PIV reconstruction

Copy `viv_piv/protocol.json`, set `data_root` and `output_root`, then run:

```bash
python -m viv_piv.validate_data --config path/to/protocol.json
python -m viv_piv.prepare --config path/to/protocol.json
python -m viv_piv.prepare_adaptive_sensor_layout \
  --config path/to/protocol.json --layout adaptive_fullfield_valid_x40y20
python -m viv_piv.prepare_observation_covariance \
  --config path/to/protocol.json --layout adaptive_fullfield_valid_x40y20 \
  --device cuda:2
python -m viv_piv.run_case --config path/to/protocol.json \
  --case 0679 --method apce --seed 0 --record-trace
python -m viv_piv.aggregate --config path/to/protocol.json
```

The public archive is identified by DOI 10.57745/HPA87O. The twelve training
cases fit the POD/DMDc library and observation covariance; the five test cases
are used only for held-out reconstruction.

## Acoustic-field reconstruction

Prepare the measured MeshRIR S1 array as an NPZ input accepted by
`acoustic_field_reconstruction.run`, then execute:

```bash
python -m acoustic_field_reconstruction.run \
  --source-npz path/to/s1_reconstruction_input.npz \
  --output results/acoustic_field --device cuda:2
```

The fixed protocol is stored in `acoustic_field_reconstruction/protocol.json`.

## Acoustic-array tracking

The tracking adapter consumes triangulated Cartesian observations, their
covariances and GPS trajectories from an authorized local data copy.

```bash
python -m acoustic_array_tracking.prepare_dual_source --help
python -m acoustic_array_tracking.run --stage track --task dual_source \
  --frontend path/to/target1/frontend --output results/acoustic_tracking \
  --method apce --seed 0 --device cuda:2
```

Preparation produces one frontend directory per target. The runner supports
DEnKF, augmented-state EnKF, BMA, PCE and APCE under the task configuration in
`acoustic_array_tracking/protocol.json`.
