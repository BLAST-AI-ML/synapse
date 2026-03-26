# Coding Agent Instructions for Synapse

## Project Overview

Synapse (**Synergistic Software Platform for AI, Physics Simulations, and Experiments**) is a modular framework for building digital twin components at Lawrence Berkeley National Laboratory. It couples experimental data, simulations, and ML models trained on combined data. The platform targets NERSC infrastructure (Spin for cloud services, Superfacility API for HPC on Perlmutter).

## Repository Structure

```
synapse/
├── dashboard/              # Trame-based web GUI application
│   ├── app.py              # Main entry point (Trame web app)
│   ├── *_manager.py        # Feature managers (model, parameters, outputs, calibration, optimization, state, sfapi, error)
│   ├── utils.py            # Shared utilities (DB access, plotting, config)
│   ├── environment.yml     # Conda dependencies for GUI
│   └── environment-lock.yml
├── ml/                     # ML training module
│   ├── train_model.py      # Main training script (GP, NN, ensemble)
│   ├── Neural_Net_Classes.py  # PyTorch neural network classes
│   ├── training_pm.sbatch  # SLURM batch script for Perlmutter
│   ├── environment.yml     # Conda dependencies for ML
│   └── environment-lock.yml
├── experiments/            # Experiment configs (cloned from private repos)
├── tests/                  # Integration tests (ML pipeline)
│   ├── test_ml_pipeline.py # Full ML training pipeline test
│   └── check_model.py      # Model checking utility
├── dashboard.Dockerfile    # Docker image for the GUI
├── ml.Dockerfile           # Docker image for ML training (CUDA 12.4)
├── publish_container.py    # Script to build & push Docker containers to NERSC registry
├── .pre-commit-config.yaml # Ruff linter/formatter hooks
└── .github/workflows/codeql.yml  # CodeQL security scanning
```

## Language and Dependencies

- **Language**: Python 3.12 (managed via Conda)
- **Dashboard dependencies**: trame (web framework), plotly, pymongo, botorch, pytorch, lume-model, sfapi_client, mlflow
- **ML dependencies**: pytorch (CUDA 12.4), gpytorch, botorch, lume-model, mlflow, pymongo, scikit-learn
- **Environment management**: Conda with `conda-lock` for reproducible environments. Each component (`dashboard/`, `ml/`) has its own `environment.yml` and `environment-lock.yml`.

## Linting and Formatting

This project uses **Ruff** for linting and formatting, configured via `.pre-commit-config.yaml`. There is no `ruff.toml` or `pyproject.toml` — Ruff runs with default rules.

```bash
# Run the linter (with auto-fix)
ruff check --fix .

# Run the formatter
ruff format .

# Run both via pre-commit (if installed)
pre-commit run --all-files
```

Always run `ruff check` and `ruff format` before committing changes.

## Building

There is no traditional build step (no `setup.py`, `pyproject.toml`, or `Makefile`). The project runs directly as Python scripts within Conda environments and is containerized via Docker for deployment.

### Docker builds (from repository root)

```bash
# Build the dashboard container
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-gui -f dashboard.Dockerfile .

# Build the ML training container
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-ml -f ml.Dockerfile .

# Automated build and publish (interactive)
python publish_container.py --gui --ml
```

## Testing

There is no pytest/unittest framework configured, but `tests/test_ml_pipeline.py` tests the full ML training pipeline (training → upload to MLflow → download from MLflow → check accuracy). It requires a local MLflow server:

```bash
# Start a local MLflow server
docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0

# Run the test from the repository root
python tests/test_ml_pipeline.py

# Optionally restrict to a specific model type or config
python tests/test_ml_pipeline.py --model NN --config_file experiments/synapse-bella-ip2
```

Dashboard validation is done manually by running the application.

## CI/CD

The only CI workflow is **CodeQL Advanced** (`.github/workflows/codeql.yml`), which runs security scanning on Python code for pushes and PRs to `main`.

## Key Architecture Patterns

### Dashboard (Trame GUI)

- Built on [Trame](https://kitware.github.io/trame/) — a Python framework for interactive web applications.
- Uses the **manager pattern**: each feature area has a dedicated `*_manager.py` class that handles its UI components and business logic.
- `state_manager.py` manages the global Trame server, state, and controller.
- Data flows through MongoDB (PyMongo) for experiment and simulation data.
- Data flows through MLflow for ML models.
- NERSC Superfacility API integration is in `sfapi_manager.py`.

### ML Training

- `train_model.py` supports three model types: Gaussian Process (GP), Neural Network (NN), and Ensemble.
- Uses PyTorch, BoTorch, and GPyTorch for model training.
- CUDA is auto-detected for GPU acceleration.
- Models are serialized and stored in an MLflow tracking server.

### Data Storage

- **MongoDB** is used for persistent data from experiments and simulations.
- **MLflow** is used for persistent data from ML models.
- Database access requires SSH tunneling to NERSC when running locally.
- Environment variables: `SF_DB_HOST` (dashboard), `SF_DB_READONLY_PASSWORD` (dashboard and ML training), `AM_SC_API_KEY` (dashboard and ML training, required when MLflow tracking_uri is AmSC).

## Common Pitfalls and Workarounds

1. **No `pyproject.toml` or `ruff.toml`**: Ruff uses default rules. Do not create these files unless the project explicitly adopts them.
2. **Conda, not pip**: Dependencies are managed via `conda` and `conda-lock`, not `pip`. Do not add `requirements.txt` or modify `pyproject.toml` for dependencies. Update `environment.yml` in the relevant component directory and regenerate the lock file.
3. **Separate environments**: The dashboard and ML components have independent Conda environments (`synapse-gui` and `synapse-ml`). Changes to dependencies must be made in the correct `environment.yml`.
4. **Docker builds from root**: Dockerfiles reference paths relative to the repository root. Always run `docker build` from the repository root directory.
5. **Limited test infrastructure**: There is no pytest/unittest framework, but `tests/test_ml_pipeline.py` can validate ML changes end-to-end (requires a local MLflow server). Always run the linter (`ruff check .`) and verify logic through code review.
6. **Experiment configs are external**: The `experiments/` directory contains cloned private repositories. These are not checked into this repository (excluded via `.gitignore`).
7. **NERSC-specific infrastructure**: Much of the deployment depends on NERSC services (Spin, Superfacility API, Perlmutter). Code changes affecting deployment or data access should be tested against NERSC services when possible.

## Making Changes

- **Python code**: Edit files directly in `dashboard/` or `ml/`. Run `ruff check --fix .` and `ruff format .` after changes.
- **Dependencies**: Edit the appropriate `environment.yml` file. Regenerate the lock file with `conda-lock`.
- **Docker**: Modify `dashboard.Dockerfile` or `ml.Dockerfile`. Rebuild with the commands above.
- **New features**: Follow the manager pattern for dashboard features — create a new `*_manager.py` file and integrate it with `app.py` and `state_manager.py`.
