# Acoustic-array source tracking

This package applies PCE/APCE to Cartesian trajectories inferred from a field
acoustic array. The state is position and velocity in local east-north-up
coordinates; each valid acoustic observation supplies a Cartesian position and
its full 3 by 3 covariance. The frozen single- and dual-source settings are in
[`protocol.json`](protocol.json).

The measured array data are read from an authorized local copy and are not
embedded in the repository. GPS positions enter only the final metric
calculation.

## Dual-source preparation

```bash
python -m acoustic_array_tracking.prepare_dual_source \
  --triangulation-root path/to/triangulation \
  --triangulation-manifest path/to/triangulation_manifest.json \
  --truth-root path/to/reference_tracks \
  --reliability-csv path/to/node_reliability.csv \
  --output results/acoustic_tracking/frontend \
  --start-s 46561 --end-s 46620
```

## Assimilation

```bash
python -m acoustic_array_tracking.run --stage track \
  --frontend results/acoustic_tracking/frontend/target1/frontend \
  --output results/acoustic_tracking --task dual_source \
  --method apce --seed 0 --device cuda:2
```

Use `--task single_source` with a single-source frontend. The runner accepts
DEnKF, augmented-state EnKF, BMA, PCE and APCE. Use `--stage aggregate` after
all requested case-method-seed runs are complete.
