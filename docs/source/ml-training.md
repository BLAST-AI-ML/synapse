# ML Training

Synapse's ML training is implemented primarily in `ml/train_model.py`.

ML models can be trained in two distinct ways:

1. Locally on your computer.

2. At NERSC, either manually or through the dashboard.

It reads the configuration and MongoDB records, trains a model, wraps it with `lume-model`, and optionally registers it in MLflow.

## Train ML Models Locally

This section describes how to train ML models locally.

### Without Docker

#### Prepare the conda environment

1. Move to the `ml/` directory.

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if it is not already installed:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Create the conda environment `synapse-ml`:
   ```bash
   conda-lock install --name synapse-ml environment-lock.yml
   ```

#### Run the training

1. Create an SSH tunnel to access the MongoDB database at NERSC (in a separate terminal):
   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```

2. Move to the `ml/` directory.

3. Set up the database settings (read-only) and the AmSC MLflow API key:
   ```bash
   export SF_DB_READONLY_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   export AM_SC_API_KEY='your_amsc_api_key_here'        # Required when MLflow tracking_uri is AmSC
   ```

4. Activate the conda environment `synapse-ml`:
   ```bash
   conda activate synapse-ml
   ```

5. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <your_model> --config_file <your_config_file>
   ```

#### Test the full train/save/load cycle: `test_ml_pipeline.py`

`tests/test_ml_pipeline.py` exercises the full ML lifecycle: training → upload to MLflow → download → accuracy check. It requires a local, empty MLflow server so it does not touch a production server.

1. Start a local MLflow server, e.g. with Docker:
   ```bash
   docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0
   ```

2. Run the test script from the root of the repository (by default this expects the MLflow server to run on `localhost:5000`):
   ```bash
   python tests/test_ml_pipeline.py
   ```

   Optionally, restrict to a specific model type or config file:
   ```bash
   python tests/test_ml_pipeline.py --model NN --config_file experiments/synapse-bella-ip2/config.yaml
   ```

   If your MLflow server is running on a different port (e.g. 5001 instead of 5000), pass it explicitly:
   ```bash
   python tests/test_ml_pipeline.py --test-mlflow-uri http://localhost:5001
   ```

### With Docker

Coming soon.

## Train ML Models at NERSC

This section describes how to train ML models at NERSC.

### Manually without Docker

#### Prepare the conda environment

1. Move to the `ml/` directory.

2. Activate your own user base conda environment:
   ```bash
   module load python
   conda activate <your_base_env>
   ```

3. Install `conda-lock` if not installed yet:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Create the conda environment `synapse-ml`:
   ```bash
   conda-lock install --name synapse-ml environment-lock.yml
   ```

#### Run the training

1. Move to the `ml/` directory.

2. Set up the database settings (read-only) and the AmSC MLflow API key:
   ```bash
   export SF_DB_READONLY_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   export AM_SC_API_KEY='your_amsc_api_key_here'        # Required when MLflow tracking_uri is AmSC
   ```

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
The Docker container is pulled from the [NERSC registry](https://registry.nersc.gov) and does not reflect any local changes you may have made to `train_model.py` unless you rebuild and redeploy the container first.
```

1. Log in to Perlmutter:
   ```bash
   ssh perlmutter-p1.nersc.gov
   ```

2. Ensure the file `$HOME/db.profile` contains the read-only database password and the AmSC MLflow API key: `export SF_DB_READONLY_PASSWORD='your_password_here'` and `export AM_SC_API_KEY='your_amsc_api_key_here'`.

3. Pull the Docker container:
   ```bash
   podman-hpc login --username $USER registry.nersc.gov
   # Password: your NERSC password without 2FA
   podman-hpc pull registry.nersc.gov/m558/superfacility/synapse-ml:latest
   ```

4. Allocate a GPU node and run the container:
   ```bash
   salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A m558
   podman-hpc run --gpu -v /etc/localtime:/etc/localtime -v $HOME/db.profile:/root/db.profile -v /path/to/config.yaml:/app/ml/config.yaml --rm -it registry.nersc.gov/m558/superfacility/synapse-ml:latest python -u /app/ml/train_model.py --test --config_file /app/ml/config.yaml --model NN
   ```
   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.

### Through the dashboard

````{warning}
When ML models are trained through the dashboard, Synapse uses NERSC's Superfacility API with the collaboration account `sf558`.
Because this is a non-interactive, non-user account, Synapse also uses a custom user to pull the image from the [NERSC registry](https://registry.nersc.gov) to Perlmutter.
The registry login credentials need to be prepared (only once) in the `$HOME` of user `sf558` (`/global/homes/s/sf558/`), in a file named `registry.profile` with the following content:
```bash
export REGISTRY_USER="robot\$m558+perlmutter-nersc-gov"
export REGISTRY_PASSWORD="..."
```
````

Connect to the [dashboard](https://bellasuperfacility.lbl.gov/) deployed at NERSC through Spin and click the `Train` button in the `ML` panel.
You need to upload valid Superfacility API credentials before you can launch simulations or train ML models directly from the dashboard.

## Model Types

Use `--model` with one of:

- `GP`: Gaussian Process.
- `NN`: single neural network.
- `ensemble_NN`: ensemble neural network. The current ensemble size is defined in `train_nn_ensemble()` in `ml/train_model.py`.

## Command

```bash
python train_model.py --config_file ../experiments/synapse-bella-ip2/config.yaml --model NN
```

Use `--test` to skip MLflow registration.

## Phases

1. Load config, variables, database records, and MLflow settings.
2. Build calibration and normalization transforms.
3. Train on simulation data.
4. Train the [calibration](experiment-configuration.md#calibration) on experimental data when available.
5. Build a `lume-model`.
6. Register to MLflow unless `--test` is set.

## MLflow Names

Registered models use:

```text
synapse-<experiment>_<model_type>
```

The MLflow experiment is:

```text
synapse-<experiment>
```

## For Maintainers

### Generate the conda environment lock file

1. Move to the directory `ml/`.

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if not installed yet:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Generate the conda environment lock file:
   ```bash
   conda-lock --file environment.yml --virtual-package-spec virtual-packages.yml --lockfile environment-lock.yml
   ```

### Build and push the Docker container to NERSC

```{warning}
Pushing a new Docker container affects ML training jobs launched from both locally deployed dashboards and the dashboard deployed at NERSC, because in both cases training runs in a Docker container pulled from the [NERSC registry](https://registry.nersc.gov).
Currently, this is the only way to test the end-to-end integration of the dashboard with the ML training workflow.
```

````{tip}
Run this workflow automatically with the Python script `publish_container.py`:
```bash
python publish_container.py --ml
```
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

2. Build the Docker image:
   ```bash
   docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-ml -f ml.Dockerfile .
   ```

#### Push the Docker container

1. Move to the root directory of the repository.

2. Log in to the [NERSC registry](https://registry.nersc.gov):
   ```bash
   docker login registry.nersc.gov
   # Username: your NERSC username
   # Password: your NERSC password without 2FA
   ```

3. Tag the Docker image:
   ```bash
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:latest
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:$(date "+%y.%m")
   ```

4. Push the Docker container:
   ```bash
   docker push -a registry.nersc.gov/m558/superfacility/synapse-ml
   ```

## References

* [Using NERSC's `registry.nersc.gov`](https://docs.nersc.gov/development/containers/registry/)
* [Podman at NERSC](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
