# Getting started

For a reproducible installation, use the pinned `environment-lock.yml` of {repo-dir}`dashboard/` or {repo-dir}`ml/` rather than the unpinned `environment.yml`.

## Run the dashboard

From {repo-dir}`dashboard/`, launch {repo}`app.py <dashboard/app.py>`:

```bash
conda-lock install --name synapse-gui environment-lock.yml
conda activate synapse-gui
export SF_DB_HOST='127.0.0.1'
export SF_DB_READONLY_PASSWORD='...'
export AM_SC_API_KEY='...'
python -u app.py --port 8080
```

For local MongoDB access, open a tunnel first:

```bash
ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
```

## Train a model

Training requires an experiment configuration.
Experiment configs are not part of this repository: clone the private repository for your experiment into {repo-dir}`experiments/` first, so that `experiments/synapse-<experiment>/config.yaml` exists.
See [Experiment configuration](experiment-configuration.md) for the expected layout.

From {repo-dir}`ml/`, run {repo}`train_model.py <ml/train_model.py>`:

```bash
conda-lock install --name synapse-ml environment-lock.yml
conda activate synapse-ml
export SF_DB_READONLY_PASSWORD='...'
export AM_SC_API_KEY='...'
python train_model.py --test --config_file ../experiments/synapse-<experiment>/config.yaml --model NN
```

## Required environment variables

- `SF_DB_HOST`: MongoDB host for the dashboard.
- `SF_DB_READONLY_PASSWORD`: read-only MongoDB password.
- `AM_SC_API_KEY`: American Science Cloud MLflow API key when the config uses that service.
