# VIV-PIV PCE/APCE transfer case

This directory contains an independent real-data case package. It reads the
17 NPZ archives from the DesignSafe-style public VIV-PIV repository on the
Super-Server and writes only manifests, reduced summaries, figures and model
outputs to the configured output directory.

The public dataset DOI is `10.57745/HPA87O` and the reported license is Etalab
2.0. Each archive contains a 1000-frame, 201 by 416 two-component PIV field,
binary cylinder mask, timestamps, cylinder displacement and component-wise
normalization limits. Coordinate values are documented as millimetres; the
README example that prints them as metres is inconsistent and is not used.

The fixed external tests are 0463, 0556, 0679, 0803 and 1359. The remaining
12 cases form the candidate-library pool. Test fields never enter POD, DMDc,
normalisation, observation-noise estimation or candidate scoring.

The operational candidate coordinate is reduced velocity parsed from the case
identifier, not a true Liu alpha, Bayesian posterior or physical parameter
estimate. The main protocol uses a fixed 20 x 40 downstream layout: 800 spatial
points and 1,600 scalar u/v observations. Cylinder displacement is supplied as
a known exogenous input and is not in the assimilation observation vector.

图5子图候选清单： [00_图5子图候选清单.md](00_图5子图候选清单.md)

A10--A30 独立诊断图： [figures/A10_A30_0679_apce_rank256/README.md](figures/A10_A30_0679_apce_rank256/README.md)

## Frozen calibrated protocol

The final external comparison uses the following training-only choices:

- POD rank 256 for the primary full-field reconstruction (rank 128 is retained
  as the kinetic-energy sensitivity control).
- Ensemble size 64; sizes 32 and 128 are reported as a sharpness/accuracy
  sensitivity study.
- Full observation covariance with shrinkage 0.05. The covariance is computed
  only from the 12 training cases; external runs with the `shr050` suffix use
  dynamic training-only construction, while the rank-256 APCE run uses a
  separately audited rank-256 archive whose manifest also records shrinkage
  0.05 and zero test cases.
- PCE initial ensemble scale 0.3 and APCE initial ensemble scale 0.1;
  process-noise scale 1.0 and state inflation 2.0 for both methods.
- Evidence window 1 frame. Causal windows of 3 and 5 frames were evaluated on
  four training pseudo-holdouts and were marginally worse, so they were not
  carried to the external tests.
- A spatial exponential taper with length (L=1D) was evaluated with the same
  shrinkage and ensemble protocol. It raised coverage to about 0.972 but
  widened intervals by about 0.54 and worsened CRPS, so the full covariance is
  retained for the primary result.

The primary rank-256 external outputs are under
`<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce/runs/rank256_stride1/`:

- PCE: `*_layout20x40_ens064_covfull_shr050.json` and matching traces.
- APCE: `*_layout20x40_ens064_covfull.json`; its covariance archive manifest
  is `models/rank256_stride1/sensor_layouts/20x40/observation_covariance_full_manifest.json`.

Across five external cases and three paired seeds, the rank-256 means are:

| method | held-out nRMSE | normalized CRPS | 90% coverage | blackout nRMSE | kinetic-energy nRMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCE | 0.205 | 0.147 | 0.956 | 0.268 | 0.032 |
| APCE | 0.204 | 0.147 | 0.960 | 0.266 | 0.030 |

These are transfer/reconstruction results, not evidence that the operational
reduced-velocity coordinate is a true cognitive variable or a Bayesian
posterior. The older runs without `shr050` used the legacy 0.18 covariance
archive and are retained only for cache/provenance auditing.

Run on the Super-Server with the project environment:

```bash
PY=python
$PY -m viv_piv_case.audit --config viv_piv_case/config.json
$PY -m viv_piv_case.prepare --config viv_piv_case/config.json
$PY -m viv_piv_case.run_case --config viv_piv_case/config.json --case 0679 --method pce --seed 0 --record-trace
$PY -m viv_piv_case.run_case --config viv_piv_case/config.json --case 0679 --method apce --seed 0 --record-trace
$PY -m viv_piv_case.aggregate --config viv_piv_case/config.json
$PY -m viv_piv_case.make_figures --config viv_piv_case/config.json
$PY -m pytest -q viv_piv_case/tests/test_viv_piv_case.py
```

The first run is an admission check. The result directory contains
`summaries/`, `figures/`, run JSON, compact trace NPZ files and a machine-
readable `leakage_audit.json`. The audit checks that train/test cases are
disjoint, test cases never enter the candidate library, weights remain finite
and normalized, and all trace values are finite. No paper TeX files are
modified by this package.

The primary figure contract is deliberately bounded: the figures test held-out
wake reconstruction, kinetic-energy reconstruction and known-input sensor
blackout forecasts, while showing shadow score separation and candidate-weight
diagnostics. A candidate-separation gate failure does not invalidate the
reconstruction metrics; it means the weights must be reported as operational
predictive evidence rather than as identifiable posterior parameters.

The NPZ archives remain read-only on the Super-Server. Only SHA-256 manifests,
training-only POD/DMDc summaries, traces, tables and figures are written under
`<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce/`.
