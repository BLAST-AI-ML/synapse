# Overview

Synapse is a modular framework for connecting experimental data, simulation data, and machine learning models for digital twin workflows.

The main source areas are:

- `dashboard/`: a Trame web application for exploring experiments, simulations, model predictions, optimization, calibration, and NERSC job controls.
- `ml/`: model training code for Gaussian Process, single Neural Network, and Neural Network ensemble models.
- `experiments/`: experiment-specific configuration and scripts, usually cloned from private repositories.
- `tests/`: integration checks for the ML pipeline.
- `docs/`: Sphinx documentation source.

The typical workflow is:

1. Add or update an experiment repository under `experiments/synapse-<name>/`.
2. Define `config.yaml` with database, MLflow, input, output, and optional calibration settings.
3. Load experiment and simulation points from MongoDB.
4. Train a model from simulation data, optionally calibrating against experimental data.
5. Register the trained model in MLflow.
6. Use the dashboard to visualize data, query the model, optimize inputs, and launch NERSC jobs.

## Services

Synapse currently assumes these external services:

- MongoDB for experiment and simulation records.
- MLflow for registered model storage.
- NERSC Spin for dashboard deployment.
- NERSC Superfacility API for Perlmutter jobs.
- NERSC container registry for dashboard and ML images.

## Repository Map

```text
dashboard/      Trame GUI and dashboard managers
ml/             ML training script, model classes, Perlmutter batch template
experiments/    Experiment configs and experiment-owned scripts
tests/          End-to-end ML pipeline helpers
docs/           Sphinx documentation source
```
