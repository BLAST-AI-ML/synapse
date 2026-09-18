# Getting started

For a reproducible installation, use the pinned `environment-lock.yml` of {repo-dir}`dashboard/` or {repo-dir}`ml/` rather than the unpinned `environment.yml`.

## Conventions

Synapse is set up per project: each deployment brings its own experiment configurations, MongoDB database, MLflow server, and NERSC project.
Where a command or a value depends on the deployment, these docs show it in two tabs:

- **General**: the command with placeholders for the values of your deployment.
- **Project Example: BELLA @ NERSC**: the same command with the values of the BELLA deployment at NERSC.

The tab you select applies to all pages while you browse in the same browser tab.

Placeholders use the following notation:

- `<...>`: a value to replace, for example `<username>`.
- `[...]`: an optional part, to replace or to omit, for example `[<namespace>/]`.
- `<section.key>`: the value of a key in your experiment's `config.yaml`, for example `<database.host>` for the `host` key of the `database` section.
  See [Experiment configuration](experiment-configuration.md).

Names such as the NERSC project `m558`, the collaboration account `sf558`, and the host `mongodb05.nersc.gov` are specific to the BELLA deployment.
Other projects use their own.

## Run the dashboard

The dashboard reads the MongoDB connection settings from the `database` section of your experiment's `config.yaml`.
If the database is only reachable through a gateway node, first open an SSH tunnel in a separate terminal:

::::{tab-set}
:sync-group: deployment

:::{tab-item} General
:sync: general

```bash
ssh -L <database.port>:<database.host>:<database.port> <username>@<gateway_host> -N
```
:::

:::{tab-item} Project Example: BELLA @ NERSC
:sync: bella-nersc

```bash
ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
```
:::
::::

Then set `database.host` to `127.0.0.1` in your local copy of `config.yaml`, and do not commit this change.

From {repo-dir}`dashboard/`, launch {repo}`app.py <dashboard/app.py>`:

::::{tab-set}
:sync-group: deployment

:::{tab-item} General
:sync: general

```bash
conda-lock install --name synapse-gui environment-lock.yml
conda activate synapse-gui
export <database.password_ro_env>='...'
export <mlflow.api_key_env>='...'
python -u app.py --port 8080
```
:::

:::{tab-item} Project Example: BELLA @ NERSC
:sync: bella-nersc

```bash
conda-lock install --name synapse-gui environment-lock.yml
conda activate synapse-gui
export SF_DB_READONLY_PASSWORD='...'
export AM_SC_API_KEY='...'
python -u app.py --port 8080
```
:::
::::

## Train a model

Training requires an experiment configuration.
Experiment configs are not part of this repository: clone the private repository for your experiment into {repo-dir}`experiments/` first, so that `experiments/synapse-<experiment>/config.yaml` exists.
See [Experiment configuration](experiment-configuration.md) for the expected layout.
If `database.port` is not `27017` and you access the database through a gateway, open a separate tunnel for training with local port `27017`: `ssh -L 27017:<database.host>:<database.port> <username>@<gateway_host> -N`.
Unlike the dashboard, {repo}`train_model.py <ml/train_model.py>` does not read `database.port` yet and always connects to the default MongoDB port.

From {repo-dir}`ml/`, run {repo}`train_model.py <ml/train_model.py>`:

::::{tab-set}
:sync-group: deployment

:::{tab-item} General
:sync: general

```bash
conda-lock install --name synapse-ml environment-lock.yml
conda activate synapse-ml
export <database.password_ro_env>='...'
export <mlflow.api_key_env>='...'
python train_model.py --test --config_file ../experiments/synapse-<experiment>/config.yaml --model NN
```
:::

:::{tab-item} Project Example: BELLA @ NERSC
:sync: bella-nersc

```bash
conda-lock install --name synapse-ml environment-lock.yml
conda activate synapse-ml
export SF_DB_READONLY_PASSWORD='...'
export AM_SC_API_KEY='...'
python train_model.py --test --config_file ../experiments/synapse-bella-ip2/config.yaml --model NN
```
:::
::::

## Required environment variables

The experiment's `config.yaml` names the environment variables that hold the credentials, so that no secret is stored in the configuration file:

::::{tab-set}
:sync-group: deployment

:::{tab-item} General
:sync: general

- `<database.password_ro_env>`: read-only MongoDB password.
- `<mlflow.api_key_env>`: MLflow API key, only needed when `mlflow.tracking_uri` is the American Science Cloud (AmSC) MLflow server, `https://mlflow.american-science-cloud.org`.
:::

:::{tab-item} Project Example: BELLA @ NERSC
:sync: bella-nersc

- `SF_DB_READONLY_PASSWORD`: read-only MongoDB password.
- `AM_SC_API_KEY`: AmSC MLflow API key.
:::
::::
