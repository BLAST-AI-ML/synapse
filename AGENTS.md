# CLAUDE.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

Synapse is a framework that couples experimental data, simulations, and machine learning models through a web-based dashboard deployed at NERSC.

## Architecture

- **`dashboard/`** — Trame (Vue.js/Flask) web application. `app.py` is the main entry point. Functionality is split into manager modules: `model_manager.py` (ML models), `parameters_manager.py` (parameter handling), `calibration_manager.py` (simulation calibration), `sfapi_manager.py` (NERSC Superfacility API), `optimization_manager.py` (Bayesian optimization), `state_manager.py`, `error_manager.py`, `outputs_manager.py`. `utils.py` handles data loading.
- **`ml/`** — PyTorch training pipeline. `train_model.py` is the main script, `Neural_Net_Classes.py` defines network architectures. Supports GP, NN, and ensemble models.
- **`experiments/`** — YAML config files for each BELLA experiment (inputs, outputs, calibration variables).
- **`dashboard.Dockerfile`** / **`ml.Dockerfile`** — Container images for GUI and ML training respectively.
- **`publish_container.py`** — Builds and pushes containers to NERSC registry.

## Key Commands

### Linting
```bash
pre-commit run --all-files    # Ruff linting + formatting
```

### Running the Dashboard Locally
```bash
conda-lock install --name synapse-gui dashboard/environment-lock.yml
conda activate synapse-gui
python -u dashboard/app.py --port 8080
```
Requires MongoDB access via `SF_DB_HOST` and `SF_DB_READONLY_PASSWORD` environment variables.

### Running ML Training Locally
```bash
conda-lock install --name synapse-ml ml/environment-lock.yml
conda activate synapse-ml
python ml/train_model.py --test --model NN --config_file config.yaml
```

### Docker
```bash
# Build
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-gui -f dashboard.Dockerfile .
docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-ml -f ml.Dockerfile .

# Publish to NERSC
python publish_container.py --gui --yes
python publish_container.py --ml --yes
```

## Key Dependencies

- **Dashboard**: Trame, Flask, PyTorch, PyMongo, Plotly, BoTorch
- **ML Training**: PyTorch, GPyTorch, BoTorch, LUME-Model, PyMongo
- **Environments**: Managed via conda-lock (`environment-lock.yml` in each directory)

## Database

MongoDB stores experimental/simulation data and ML models. Dashboard uses read-only access (`SF_DB_READONLY_PASSWORD`), training uses admin access (`SF_DB_ADMIN_PASSWORD`).

## Deployment

- Dashboard deployed at NERSC Spin (bellasuperfacility.lbl.gov)
- ML training runs on Perlmutter HPC, triggered via Superfacility API from the dashboard
- Container registry: `registry.nersc.gov/m558/superfacility/`
