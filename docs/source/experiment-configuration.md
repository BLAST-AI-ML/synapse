# Experiment configuration

An experiment is a directory named `experiments/synapse-<experiment>/`.
The dashboard strips `synapse-` and uses the rest as the experiment identifier.

Clone the private repository for your experiment into the {repo-dir}`experiments/` directory.

Each experiment should provide:

- `config.yaml`
- optional `simulation_scripts/`
- optional `experiment_scripts/`

## Required configuration sections

- `experiment`: collection and model namespace, for example `bella-ip2`.
- `database`: MongoDB connection and credential environment variables.
- `mlflow`: tracking URI and optional API key environment variable.
- `execution_mode`: ML training and simulation mode hints.
- `inputs`: scalar variables with `name`, `type`, `default`, and `value_range`.
- `outputs`: scalar variables with `name` and `type`.

## Database and MLflow settings

The `database` and `mlflow` sections hold the connection settings of a deployment.
Secrets are not stored in `config.yaml`: the keys ending in `_env` name the environment variables that hold them.

- `database.host`, `database.port`: the MongoDB server.
  When you access the database through an SSH tunnel, set `database.host` to `127.0.0.1` in your local copy of `config.yaml` (see [Getting started](getting-started.md#run-the-dashboard)).
- `database.name`: the database.
- `database.auth`: the authentication database of the user.
- `database.username_ro`: the read-only user.
- `database.password_ro_env`: the environment variable that holds the password of the read-only user.
- `mlflow.tracking_uri`: the MLflow tracking server.
  Without it, the dashboard cannot load models and the training script does not register them.
- `mlflow.api_key_env`: the environment variable that holds the MLflow API key, only used when `mlflow.tracking_uri` is the AmSC MLflow server, `https://mlflow.american-science-cloud.org`.

::::{tab-set}
:sync-group: deployment

:::{tab-item} General
:sync: general

```yaml
experiment: "<experiment>"

database:
  host: "<database_host>"
  port: <database_port>
  name: "<database_name>"
  auth: "<authentication_database>"
  username_ro: "<read_only_user>"
  password_ro_env: "<PASSWORD_ENV_VAR>"

mlflow:
  tracking_uri: "<mlflow_tracking_uri>"
  api_key_env: "<API_KEY_ENV_VAR>"
```
:::

:::{tab-item} Project Example: BELLA @ NERSC
:sync: bella-nersc

```yaml
experiment: "bella-ip2"

database:
  host: "mongodb05.nersc.gov"
  port: 27017
  # name, auth, username_ro: see the experiment repository
  password_ro_env: "SF_DB_READONLY_PASSWORD"

mlflow:
  tracking_uri: "https://mlflow.american-science-cloud.org"
  api_key_env: "AM_SC_API_KEY"
```
:::
::::

## Simulation calibration

`simulation_calibration` maps simulation variable names to experimental variable names:

```yaml
simulation_calibration:
  input1:
    name: "simulation_variable"
    unit: "unit"
    depends_on: "experimental_variable"
    alpha_guess: 1.0
    alpha_uncertainty: 0.1
    beta_guess: 0.0
    beta_uncertainty: 0.0
```

Dashboard display uses:

```text
experimental = simulation / alpha + beta
```

Simulation launch uses:

```text
simulation = alpha * (experimental - beta)
```

These are inverse conversions: display maps simulation to experimental units, while launch maps dashboard parameters back to simulation units.

## Add an experiment

1. Clone or create `experiments/synapse-<experiment>/`.
2. Add `config.yaml`.
3. Ensure MongoDB fields match the configured input and output variable names.
4. Add `simulation_scripts/` only if dashboard launch is needed.
5. Train and register a model if dashboard predictions are needed.
