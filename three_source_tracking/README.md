# Three-source acoustic tracking

Acoustic-source localization and trajectory tracking from distributed microphone-array measurements.

## Workflow

1. `triple_frontend.py`: extract azimuth and elevation candidates.
2. `provisional_triple_association.py`: associate candidate peaks within array nodes.
3. `joint3d_multinode_tracker.py`: estimate three-dimensional source positions across nodes.
4. `build_cartesian_inputs.py`: prepare Cartesian tracking observations.
5. `run_cartesian_pce_apce.py`: run the candidate-dynamics tracking methods.

The pipeline accepts measured-array recordings and reference trajectories through command-line paths. The reference tracks are used for calibration and evaluation.

## Commands

```bash
python triple_frontend.py --help
python joint3d_multinode_tracker.py --help
python build_cartesian_inputs.py --help
python run_cartesian_pce_apce.py --stage track --help
```
