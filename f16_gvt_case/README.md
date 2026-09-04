# F-16 GVT 7.3 Hz nonlinear-modal pilot

This package is an isolated, auditable pilot for the public full-scale F-16
ground-vibration-test benchmark. It uses measured shaker force as input,
accelerations 1--2 as online observations, and acceleration 3 as a held-out
response. The first-stage model is restricted to the nonlinear wing-torsion
mode near 7.3 Hz.

Protocol boundaries:

- FullMSine Level 1 identifies the stable linear background model.
- Levels 3, 5 and 7 estimate amplitude-dependent modal uncertainty.
- Levels 2, 4 and 6 are never used for model or hyperparameter selection.
- SineSweep and SpecialOddMSine are diagnostics only.
- No physical ground-truth alpha is claimed.
- The 72-run smoke stops at the admission report; it does not automatically
  launch the 20-seed formal matrix or modify the manuscript.

Current status: the first four-state modal-ROM pilot completed `72/72` runs
but failed the preregistered admission gate. The candidate oracle did not
improve over the fixed background model, held-out nRMSE exceeded `0.50` at all
three validation levels, and blackout conclusions were not directionally
consistent. See `F16_GVT_PILOT_ADMISSION_REPORT.md`. A restart requires a new
physics-informed clearance/friction or validated LPV ROM; temperature retuning
of the rejected model is out of scope.

Dataset: https://doi.org/10.4121/12954911.v1
