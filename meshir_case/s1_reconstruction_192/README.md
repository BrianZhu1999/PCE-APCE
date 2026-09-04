# MeshRIR S1: 192-point three-dimensional reconstruction

This is an independent Stage A reconstruction package. It does not modify the
manuscript or the earlier `7 x 7 x 3` smoke results.

## Information split

- 128 boundary points: geometry-only farthest-point sampling on the six faces;
- 64 interior points: geometry-only farthest-point sampling with coverage of all
  seven interior z-layers;
- 2,463 held-out interior points: primary reconstruction evaluation;
- 1,314 held-out boundary points: boundary-completion diagnostic.

The boundary field is interpolated using only the 128 selected boundary
measurements. The 64 interior measurements are used only for state assimilation.
The sound speed is fitted from selected boundary direct-arrival times and then
fixed for the Stage A dynamics. No held-out point is used for fitting or
selection.

## Stage A/B smoke

The first fixed-speed run compared a 192-point scattered interpolation baseline
with a DEnKF reconstruction using the full `21 x 21 x 9` finite-difference wave
state. A follow-up single-seed smoke also runs BMA, PCE and APCE with candidate
boundary-completion models. The candidate coordinate is the boundary completion
model:

- candidate 0: piecewise-linear boundary completion;
- candidates 1--7: Gaussian boundary smoothers with length scales
  `0.08, 0.11, 0.14, 0.17, 0.20, 0.24, 0.30 m`.

This coordinate is not Liu alpha and not sound speed. Sound speed is calibrated
from the selected boundary direct-arrival times and then fixed.

The authoritative result bundle is on the Super-Server under
`meshir_s1_reconstruction_192_20260819/`. The complete four-method smoke with
the linear boundary candidate included is under
`stage_b_boundary_closure_with_linear/`.
