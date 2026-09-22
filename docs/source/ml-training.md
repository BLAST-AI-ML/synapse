# ML training

Synapse's ML training is implemented primarily in {repo}`ml/train_model.py`.
It reads the configuration and MongoDB records, trains a model, wraps it with `lume-model`, and optionally registers it in MLflow.

ML models can be trained in two distinct ways:

1. Locally on your computer.

2. At NERSC, either manually or through the dashboard.

## Train ML models locally

This section describes how to train ML models locally.

### Without Docker

#### Prepare the conda environment

1. Move to the {repo-dir}`ml/` directory.

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if it is not already installed:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Create the conda environment `synapse-ml` from {repo}`environment-lock.yml <ml/environment-lock.yml>`:
   ```bash
   conda-lock install --name synapse-ml environment-lock.yml
   ```

#### Run the training

1. If the computer running the training cannot directly reach the host specified by `database.host`, create an SSH tunnel to the MongoDB database through a gateway node in a separate terminal, using `database.host` and `database.port` from your experiment's `config.yaml`:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   ssh -L 27017:<database.host>:<database.port> <username>@<gateway_host> -N
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```
   :::
   ::::

   ```{note}
   The local port is 27017 because {repo}`train_model.py <ml/train_model.py>` does not read `database.port` yet and always connects to the default MongoDB port.
   ```

2. If you created the SSH tunnel, set `database.host` to `127.0.0.1` in your local copy of `config.yaml`, so that the training connects through it, but do not commit this change.
   ```yaml
   database:
     host: "127.0.0.1"
   ```

3. Move to the {repo-dir}`ml/` directory.

4. Set up the read-only database password and the MLflow API key:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   export <database.password_ro_env>='your_password_here'  # Use SINGLE quotes around the password!
   export <mlflow.api_key_env>='your_api_key_here'         # Required when MLflow tracking_uri is AmSC
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   export SF_DB_READONLY_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   export AM_SC_API_KEY='your_amsc_api_key_here'        # Required when MLflow tracking_uri is AmSC
   ```
   :::
   ::::

5. Activate the conda environment `synapse-ml`:
   ```bash
   conda activate synapse-ml
   ```

6. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <your_model> --config_file <your_config_file>
   ```

#### Test the full train/save/load cycle

{repo}`tests/test_ml_pipeline.py` exercises the full ML lifecycle: training, upload to MLflow, download, and accuracy check.
It requires a local, empty MLflow server so it does not touch a production server.

1. Start a local MLflow server, e.g. with Docker:
   ```bash
   docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0
   ```

2. Run the test script from the root of the repository (by default this expects the MLflow server to run on `localhost:5000`):
   ```bash
   python tests/test_ml_pipeline.py
   ```

   Optionally, restrict to a specific model type or config file:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   python tests/test_ml_pipeline.py --model NN --config_file experiments/synapse-<experiment>/config.yaml
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   python tests/test_ml_pipeline.py --model NN --config_file experiments/synapse-<experiment>/config.yaml
   ```
   :::
   ::::

   If your MLflow server is running on a different port (e.g. 5001 instead of 5000), pass it explicitly:
   ```bash
   python tests/test_ml_pipeline.py --test-mlflow-uri http://localhost:5001
   ```

### With Docker

Coming soon.

## Train ML models at NERSC

This section describes how to train ML models at NERSC.

### Manually without Docker

#### Prepare the conda environment

1. Move to the {repo-dir}`ml/` directory.

2. Activate your own user base conda environment:
   ```bash
   module load python
   conda activate <your_base_env>
   ```

3. Install `conda-lock` if it is not already installed:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Create the conda environment `synapse-ml` from {repo}`environment-lock.yml <ml/environment-lock.yml>`:
   ```bash
   conda-lock install --name synapse-ml environment-lock.yml
   ```

#### Run the training

1. Move to the {repo-dir}`ml/` directory.

2. Set up the read-only database password and the MLflow API key:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   export <database.password_ro_env>='your_password_here'  # Use SINGLE quotes around the password!
   export <mlflow.api_key_env>='your_api_key_here'         # Required when MLflow tracking_uri is AmSC
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   export SF_DB_READONLY_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   export AM_SC_API_KEY='your_amsc_api_key_here'        # Required when MLflow tracking_uri is AmSC
   ```
   :::
   ::::

3. Activate the conda environment `synapse-ml`:
   ```bash
   module load python
   conda activate synapse-ml
   ```

4. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <your_model> --config_file <your_config_file>
   ```

### Manually with Docker

```{warning}
The Docker image is pulled from the [NERSC registry](https://registry.nersc.gov) and does not reflect any local changes you may have made to {repo}`train_model.py <ml/train_model.py>` unless you rebuild and redeploy the image first.
```

1. Log in to Perlmutter:
   ```bash
   ssh perlmutter-p1.nersc.gov
   ```

2. Ensure the file `$HOME/db-podman.profile` contains the read-only database password and the MLflow API key.
   The file is passed to the container with `--env-file`, so write one `NAME=value` per line, without `export` and without quotes, which would become part of the value:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```text
   <database.password_ro_env>=your_password_here
   <mlflow.api_key_env>=your_api_key_here
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```text
   SF_DB_READONLY_PASSWORD=your_password_here
   AM_SC_API_KEY=your_amsc_api_key_here
   ```
   :::
   ::::

   Run `chmod 600 $HOME/db-podman.profile` so that only you can read it.

3. Pull the Docker image from the registry path of your NERSC project:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   podman-hpc login --username $USER registry.nersc.gov
   # Password: your NERSC password without 2FA
   podman-hpc pull registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-ml:latest
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   podman-hpc login --username $USER registry.nersc.gov
   # Password: your NERSC password without 2FA
   podman-hpc pull registry.nersc.gov/m558/superfacility/synapse-ml:latest
   ```
   :::
   ::::

4. Allocate a GPU node, charged to your NERSC project, and run the container:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A <nersc_project>
   podman-hpc run --gpu -v /etc/localtime:/etc/localtime --env-file $HOME/db-podman.profile -v <your_config_file>:/app/ml/config.yaml --rm -it registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-ml:latest python -u /app/ml/train_model.py --test --config_file /app/ml/config.yaml --model NN
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A m558
   podman-hpc run --gpu -v /etc/localtime:/etc/localtime --env-file $HOME/db-podman.profile -v <your_config_file>:/app/ml/config.yaml --rm -it registry.nersc.gov/m558/superfacility/synapse-ml:latest python -u /app/ml/train_model.py --test --config_file /app/ml/config.yaml --model NN
   ```
   :::
   ::::

   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.

### Through the dashboard

````{warning}
When ML models are trained through the dashboard, Synapse submits the batch script {repo}`ml/training_pm.sbatch` through NERSC's Superfacility API, as the user of the Superfacility API client (see [Generate Superfacility API credentials](dashboard.md#generate-superfacility-api-credentials)).
The batch job reads two files from the `$HOME` of that user, which need to be prepared once:

- `db-podman.profile`: the read-only database password and the MLflow API key, in the format described in [Manually with Docker](#manually-with-docker).
- `registry.profile`: the login credentials used to pull the image from the [NERSC registry](https://registry.nersc.gov) to Perlmutter.
  A collaboration account cannot log in to the registry interactively, so use a robot account of your registry project:

  ::::{tab-set}
  :sync-group: deployment

  :::{tab-item} General
  :sync: general

  ```bash
  export REGISTRY_USER="robot\$<nersc_project>+<robot_name>"
  export REGISTRY_PASSWORD="..."
  ```
  :::

  :::{tab-item} Project Example: BELLA @ NERSC
  :sync: bella-nersc

  In the `$HOME` of the collaboration account `sf558`, `/global/homes/s/sf558/`:
  ```bash
  export REGISTRY_USER="robot\$m558+perlmutter-nersc-gov"
  export REGISTRY_PASSWORD="..."
  ```
  :::
  ::::
````

```{note}
{repo}`ml/training_pm.sbatch` and {repo}`dashboard/model_manager.py` hardcode values of the BELLA deployment: the NERSC project `m558`, the `realtime` QoS, the image `registry.nersc.gov/m558/superfacility/synapse-ml`, and the directory `/global/cfs/cdirs/m558/superfacility/model_training/` for the configuration file and the job logs.
Other projects need to adapt these values before training ML models through the dashboard.
```

Connect to the dashboard deployed for your project at NERSC through Spin (see [Run the dashboard at NERSC](dashboard.md#run-the-dashboard-at-nersc)) and click the `Train` button in the `ML` panel.
You need to upload valid Superfacility API credentials before you can launch simulations or train ML models directly from the dashboard.

## Model types

Use `--model` with one of:

- `GP`: Gaussian Process.
- `NN`: single neural network.
- `ensemble_NN`: ensemble neural network.
  The current ensemble size is defined in `train_nn_ensemble()` in {repo}`ml/train_model.py`.

## Training command

```bash
python train_model.py --config_file ../experiments/synapse-<experiment>/config.yaml --model NN
```

Use `--test` to skip MLflow registration.

## What the training script does

1. Load config, variables, database records, and MLflow settings.
2. Build calibration and normalization transforms.
3. Train on simulation data.
   The script logs this step as `Phase 1`.
4. Train the [calibration](experiment-configuration.md#simulation-calibration) on experimental data when available.
   The script logs this step as `Phase 2`, and skips it when no experimental data is found.
5. Build a `lume-model`.
6. Register to MLflow, unless `--test` is set or the configuration file has no `mlflow.tracking_uri`.

```{note}
The script's own `Phase 1` and `Phase 2` log messages refer to steps 3 and 4 above, not to steps 1 and 2.
```

## MLflow model and experiment names

Registered models use:

```text
synapse-<experiment>_<model_type>
```

The MLflow experiment is:

```text
synapse-<experiment>
```

## For maintainers

### Generate the conda environment lock file

1. Move to the directory {repo-dir}`ml/`.

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if it is not already installed:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Generate the conda environment lock file from {repo}`environment.yml <ml/environment.yml>` and {repo}`virtual-packages.yml <ml/virtual-packages.yml>`:
   ```bash
   conda-lock --file environment.yml --virtual-package-spec virtual-packages.yml --lockfile environment-lock.yml
   ```

### Build and push the Docker image to NERSC

```{warning}
Pushing a new Docker image affects ML training jobs launched from both locally deployed dashboards and the dashboard deployed at NERSC, because in both cases training runs in a container started from the image pulled from the [NERSC registry](https://registry.nersc.gov).
Currently, this is the only way to test the end-to-end integration of the dashboard with the ML training workflow.
```

````{tip}
Run this workflow automatically with the Python script {repo}`publish_container.py`:
```bash
python publish_container.py --ml
```
The script pushes to `registry.nersc.gov/m558/superfacility`, which is hardcoded for the BELLA deployment.
For other projects, follow the steps below.
````

````{tip}
Prune old, unused images periodically to free up space on your machine:
```bash
docker system prune -a
```
````

#### Build the Docker image

````{important}
Ensure you have Docker version 29 or later [installed](https://docs.docker.com/engine/install/):
```bash
docker --version
```
````

1. Move to the root directory of the repository.

2. Build the Docker image defined in {repo}`ml.Dockerfile`:
   ```bash
   docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-ml -f ml.Dockerfile .
   ```

#### Push the Docker image

1. Move to the root directory of the repository.

2. Log in to the [NERSC registry](https://registry.nersc.gov):
   ```bash
   docker login registry.nersc.gov
   # Username: your NERSC username
   # Password: your NERSC password without 2FA
   ```

3. Tag the Docker image with the registry path of your NERSC project:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   docker tag synapse-ml:latest registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-ml:latest
   docker tag synapse-ml:latest registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-ml:$(date "+%y.%m")
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:latest
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:$(date "+%y.%m")
   ```
   :::
   ::::

4. Push the Docker image:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   docker push -a registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-ml
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   docker push -a registry.nersc.gov/m558/superfacility/synapse-ml
   ```
   :::
   ::::

## References

* [Using NERSC's `registry.nersc.gov`](https://docs.nersc.gov/development/containers/registry/)
* [Podman at NERSC](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
