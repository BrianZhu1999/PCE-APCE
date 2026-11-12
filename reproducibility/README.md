# Controlled-feedback and statistical reproduction

Scripts for the scalar feedback experiment and for recomputing pharmacokinetic (PK) and FitzHugh–Nagumo (FHN) statistics.

## Requirements

Python 3.12 and the packages in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Run

From this directory:

```bash
python run_feedback.py --config config.json --output generated_controlled
python recompute_fhn_statistics.py --input inputs/fhn/budget_per_run.csv --output generated_statistics
python build_figures.py --input-root inputs --controlled-root generated_controlled --output generated_figures
python verify_reproduction.py --controlled-root generated_controlled --statistics-root generated_statistics --figure-root generated_figures
```

The feedback simulation uses `config.json`. The FHN, PK and figure scripts additionally require the per-realization CSV inputs and reference tables listed in `inputs/data_sources.json`. Output directories are specified by the commands above.

The plots correspond to the scalar-feedback analysis in Fig. 1 and the PK and FHN comparisons in Fig. 2. The numerical comparisons are recorded in `reproduction_check.json`.
