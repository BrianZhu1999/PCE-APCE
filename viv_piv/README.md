# VIV-PIV wake reconstruction

This package transfers PCE/APCE to sparse two-component PIV measurements of a
vortex-induced-vibration wake. The public dataset contains 17 reduced-velocity
conditions, each with 1,000 velocity frames on a 201 by 416 grid, a cylinder
mask, timestamps and measured cylinder displacement.

Dataset: [DOI 10.57745/HPA87O](https://doi.org/10.57745/HPA87O), Etalab 2.0.

## Protocol

- Training conditions: 0432, 0494, 0525, 0587, 0618, 0648, 0741, 0865,
  0926, 0988, 1112 and 1482.
- Held-out conditions: 0463, 0556, 0679, 0803 and 1359.
- POD rank: 256; ensemble size: 64.
- Sparse layout: 800 spatial locations, yielding 1,600 scalar velocity
  observations per observed frame.
- Cylinder displacement is a known exogenous input to DMDc. The assimilation
  observation contains only the selected velocity components.
- POD, DMDc and observation covariance use training conditions only.

The complete numerical values are stored in [`protocol.json`](protocol.json).

## Run

Copy `protocol.json` and replace its two path placeholders.

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

`run_matrix` provides the same interface for a case-method-seed matrix:

```bash
python -m viv_piv.run_matrix --config path/to/protocol.json \
  --methods pce,apce,aug_enkf,bma --cases 0463,0556,0679,0803,1359 \
  --seeds 0,1,2,3,4 --gpus 2,3 --record-trace
```
